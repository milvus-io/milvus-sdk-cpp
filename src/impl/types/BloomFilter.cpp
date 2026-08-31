// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// The bit layout here is bit-identical to Arrow C++'s parquet::BlockSplitBloomFilter, and
// therefore to the parquet-format BloomFilter.md spec, wrapped in the Milvus MBF1 envelope.
// See docs/design-docs/design_docs/20260707-bloom-filter-expression.md in the milvus repo.
//
// MBF1 envelope (all integers little-endian):
//
//   offset  size  field
//   0       4     magic "MBF1"
//   4       2     version      (= 1)
//   6       2     algo         (1 = parquet_sbbf_xxh64)
//   8       8     n_declared   (informational)
//   16      8     fpr_declared (float64, informational)
//   24      4     num_blocks   (body length must equal num_blocks * 32)
//   28      1     domains      (bitmask: 1 = int64, 2 = utf8)
//   29      3     reserved     (must be 0)
//   32      ...   body: num_blocks blocks of eight little-endian uint32 words

#include "milvus/types/BloomFilter.h"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "../utils/LittleEndian.h"

namespace milvus {

namespace {

constexpr uint32_t kFilterHeaderSize = 32;
constexpr uint32_t kBytesPerBlock = 32;
constexpr uint32_t kWordsPerBlock = 8;
constexpr uint8_t kDomainInt64 = 1;
constexpr uint8_t kDomainUTF8 = 2;
constexpr uint32_t kMinFilterBytes = 32;
constexpr uint32_t kMaxFilterBytes = 128 * 1024 * 1024;

// Fixed by the parquet-format spec, mirrored from Arrow C++'s BlockSplitBloomFilter::SALT.
constexpr std::array<uint32_t, kWordsPerBlock> kSalt = {0x47b6137b, 0x44974d91, 0x8824ad5b, 0xa2b7289d,
                                                        0x705495c7, 0x2df1424b, 0x9efc4947, 0x5c6bfb31};

constexpr uint64_t kPrime64_1 = 11400714785074694791ULL;
constexpr uint64_t kPrime64_2 = 14029467366897019727ULL;
constexpr uint64_t kPrime64_3 = 1609587929392839161ULL;
constexpr uint64_t kPrime64_4 = 9650029242287828579ULL;
constexpr uint64_t kPrime64_5 = 2870177450012600261ULL;

inline uint64_t
Rotl64(uint64_t value, int count) {
    return (value << count) | (value >> (64 - count));
}

inline uint64_t
Round64(uint64_t acc, uint64_t value) {
    acc += value * kPrime64_2;
    acc = Rotl64(acc, 31);
    return acc * kPrime64_1;
}

inline uint64_t
MergeRound64(uint64_t acc, uint64_t value) {
    acc ^= Round64(0, value);
    return acc * kPrime64_1 + kPrime64_4;
}

inline uint64_t
Avalanche(uint64_t h) {
    h ^= h >> 33;
    h *= kPrime64_2;
    h ^= h >> 29;
    h *= kPrime64_3;
    return h ^ (h >> 32);
}

// Reads are byte-wise rather than a reinterpret_cast so the body layout stays little-endian
// on any host and the loads stay free of alignment and strict-aliasing assumptions.
/**
 * XXH64 with seed 0 -- the hash the SBBF spec mandates. Implemented here rather than pulled
 * in as a dependency: the SDK's conan closure carries no xxhash, and this is the whole of it.
 */
uint64_t
XXH64(const uint8_t* data, size_t length) {
    size_t index = 0;
    uint64_t result;

    if (length >= 32) {
        uint64_t v1 = kPrime64_1 + kPrime64_2;
        uint64_t v2 = kPrime64_2;
        uint64_t v3 = 0;
        uint64_t v4 = 0ULL - kPrime64_1;
        const size_t limit = length - 32;
        while (index <= limit) {
            v1 = Round64(v1, ReadU64LE(data + index));
            v2 = Round64(v2, ReadU64LE(data + index + 8));
            v3 = Round64(v3, ReadU64LE(data + index + 16));
            v4 = Round64(v4, ReadU64LE(data + index + 24));
            index += 32;
        }
        result = Rotl64(v1, 1) + Rotl64(v2, 7) + Rotl64(v3, 12) + Rotl64(v4, 18);
        result = MergeRound64(result, v1);
        result = MergeRound64(result, v2);
        result = MergeRound64(result, v3);
        result = MergeRound64(result, v4);
    } else {
        result = kPrime64_5;
    }

    result += static_cast<uint64_t>(length);

    while (index + 8 <= length) {
        result ^= Round64(0, ReadU64LE(data + index));
        result = Rotl64(result, 27) * kPrime64_1 + kPrime64_4;
        index += 8;
    }
    if (index + 4 <= length) {
        result ^= static_cast<uint64_t>(ReadU32LE(data + index)) * kPrime64_1;
        result = Rotl64(result, 23) * kPrime64_2 + kPrime64_3;
        index += 4;
    }
    while (index < length) {
        result ^= static_cast<uint64_t>(data[index]) * kPrime64_5;
        result = Rotl64(result, 11) * kPrime64_1;
        index++;
    }
    return Avalanche(result);
}

/**
 * XXH64 over an int64's 8-byte little-endian encoding, without materialising those bytes.
 *
 * Specialisation of XXH64() for the one input length the int64 domain ever produces: the
 * 32-byte stripe loop and both the 4-byte and 1-byte tails are unreachable, leaving a single
 * 8-byte lane. Reinterpreting the value as uint64 is exactly its little-endian two's
 * complement encoding read back, so the pack/unpack round trip drops out as well.
 */
inline uint64_t
XXH64Int64(int64_t value) {
    uint64_t acc = static_cast<uint64_t>(value) * kPrime64_2;
    acc = Rotl64(acc, 31);
    acc *= kPrime64_1;
    uint64_t result = (kPrime64_5 + 8) ^ acc;
    result = Rotl64(result, 27) * kPrime64_1 + kPrime64_4;
    return Avalanche(result);
}

uint32_t
NextPowerOfTwo(uint32_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

/**
 * Mirrors Arrow's BlockSplitBloomFilter::OptimalNumOfBytes: m = -8n / ln(1 - fpp^(1/8)),
 * rounded up to the next power of two and clamped to [kMinFilterBytes, kMaxFilterBytes]. The
 * result is always a power of two and a multiple of kBytesPerBlock.
 */
uint32_t
OptimalNumOfBytes(uint64_t ndv, double fpp) {
    constexpr uint32_t kMinBits = kMinFilterBytes << 3;
    constexpr uint32_t kMaxBits = kMaxFilterBytes << 3;
    const double m = -8.0 * static_cast<double>(ndv) / std::log(1.0 - std::pow(fpp, 1.0 / 8.0));

    uint32_t num_bits;
    if (m < 0 || m > static_cast<double>(kMaxBits)) {
        num_bits = kMaxBits;
    } else {
        num_bits = static_cast<uint32_t>(m);
    }
    if (num_bits < kMinBits) {
        num_bits = kMinBits;
    }
    if ((num_bits & (num_bits - 1)) != 0) {
        num_bits = NextPowerOfTwo(num_bits);
    }
    if (num_bits > kMaxBits) {
        num_bits = kMaxBits;
    }
    return num_bits >> 3;
}

bool
IsValidFPR(double fpr) {
    return !std::isnan(fpr) && fpr >= BloomFilterMinFPR && fpr <= BloomFilterMaxFPR;
}

std::string
FPRErrorMessage(double fpr) {
    return "Bloom filter fpr " + std::to_string(fpr) + " is out of range [" + std::to_string(BloomFilterMinFPR) + ", " +
           std::to_string(BloomFilterMaxFPR) + "]";
}

}  // namespace

BloomFilterBuilder::BloomFilterBuilder(uint64_t n, double fpr) {
    if (!IsValidFPR(fpr)) {
        throw std::runtime_error(FPRErrorMessage(fpr));
    }
    const uint32_t num_bytes = OptimalNumOfBytes(n, fpr);
    buf_.assign(kFilterHeaderSize + num_bytes, 0);
    num_blocks_ = num_bytes / kBytesPerBlock;
    n_declared_ = n;
    fpr_ = fpr;
}

void
BloomFilterBuilder::addHash(uint64_t hash) {
    // Multiply-shift block reduction, as Arrow does. num_blocks_ is at most 2^22, so the
    // product cannot overflow.
    const auto block = static_cast<uint32_t>(((hash >> 32) * num_blocks_) >> 32);
    uint8_t* blk = buf_.data() + kFilterHeaderSize + static_cast<size_t>(block) * kBytesPerBlock;
    const auto key = static_cast<uint32_t>(hash);

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    // The stored words are already in host order, so the whole 32-byte block can be loaded,
    // OR-ed and stored as a unit. memcpy is the aliasing-safe spelling of that and compilers
    // turn it into a pair-load / vector-OR / pair-store: on arm64 the loop below becomes one
    // ldp + two orr.16b + stp. Doing the same work through byte-wise accessors defeats the
    // vectoriser and costs about 4x on a 10M-member build, which is why this path exists.
    std::array<uint32_t, kWordsPerBlock> words;
    std::memcpy(words.data(), blk, kBytesPerBlock);
    for (uint32_t i = 0; i < kWordsPerBlock; i++) {
        words[i] |= 1u << ((key * kSalt[i]) >> 27);
    }
    std::memcpy(blk, words.data(), kBytesPerBlock);
#else
    // Big-endian or unknown byte order: go through the explicit little-endian accessors so
    // the body keeps the layout the spec mandates. Correctness first; these hosts are rare.
    for (uint32_t i = 0; i < kWordsPerBlock; i++) {
        const uint32_t mask = 1u << ((key * kSalt[i]) >> 27);
        WriteU32LE(blk + i * 4, ReadU32LE(blk + i * 4) | mask);
    }
#endif
}

BloomFilterBuilder&
BloomFilterBuilder::AddInt64(int64_t value) {
    domains_ |= kDomainInt64;
    addHash(XXH64Int64(value));
    return *this;
}

BloomFilterBuilder&
BloomFilterBuilder::AddString(const std::string& value) {
    domains_ |= kDomainUTF8;
    addHash(XXH64(reinterpret_cast<const uint8_t*>(value.data()), value.size()));
    return *this;
}

BloomFilterBuilder&
BloomFilterBuilder::AddInt64s(const std::vector<int64_t>& values) {
    if (values.empty()) {
        return *this;
    }
    domains_ |= kDomainInt64;
    for (const auto value : values) {
        addHash(XXH64Int64(value));
    }
    return *this;
}

BloomFilterBuilder&
BloomFilterBuilder::AddStrings(const std::vector<std::string>& values) {
    if (values.empty()) {
        return *this;
    }
    domains_ |= kDomainUTF8;
    for (const auto& value : values) {
        addHash(XXH64(reinterpret_cast<const uint8_t*>(value.data()), value.size()));
    }
    return *this;
}

uint8_t
BloomFilterBuilder::Domains() const {
    return domains_;
}

uint32_t
BloomFilterBuilder::NumBlocks() const {
    return num_blocks_;
}

std::vector<uint8_t>
BloomFilterBuilder::Build() const {
    std::vector<uint8_t> blob = buf_;
    std::memcpy(blob.data(), "MBF1", 4);
    WriteU16LE(blob.data() + 4, 1);  // version
    WriteU16LE(blob.data() + 6, 1);  // algo = parquet_sbbf_xxh64
    WriteU64LE(blob.data() + 8, n_declared_);
    uint64_t fpr_bits = 0;
    std::memcpy(&fpr_bits, &fpr_, sizeof(fpr_bits));
    WriteU64LE(blob.data() + 16, fpr_bits);
    WriteU32LE(blob.data() + 24, num_blocks_);
    blob[28] = domains_;
    // blob[29..31] stays zero (reserved).
    return blob;
}

nlohmann::json
BloomFilterBuilder::BuildTemplate() const {
    return nlohmann::json::binary(Build());
}

Status
BloomFilterTemplate(const std::vector<int64_t>& members, double fpr, nlohmann::json& output) {
    if (!IsValidFPR(fpr)) {
        return {StatusCode::INVALID_ARGUMENT, FPRErrorMessage(fpr)};
    }
    BloomFilterBuilder builder(members.size(), fpr);
    builder.AddInt64s(members);
    output = builder.BuildTemplate();
    return Status::OK();
}

Status
BloomFilterTemplate(const std::vector<std::string>& members, double fpr, nlohmann::json& output) {
    if (!IsValidFPR(fpr)) {
        return {StatusCode::INVALID_ARGUMENT, FPRErrorMessage(fpr)};
    }
    BloomFilterBuilder builder(members.size(), fpr);
    builder.AddStrings(members);
    output = builder.BuildTemplate();
    return Status::OK();
}

Status
EstimateBloomFilterSize(uint64_t n, double fpr, uint64_t& output) {
    if (!IsValidFPR(fpr)) {
        return {StatusCode::INVALID_ARGUMENT, FPRErrorMessage(fpr)};
    }
    output = kFilterHeaderSize + OptimalNumOfBytes(n, fpr);
    return Status::OK();
}

}  // namespace milvus
