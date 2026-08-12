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

#pragma once

#include <cstdint>
#include <milvus/thirdparty/nlohmann/json.hpp>
#include <string>
#include <vector>

#include "milvus/Export.h"
#include "milvus/Status.h"

namespace milvus {

/**
 * @brief Lowest accepted false-positive rate for a bloom filter.
 */
constexpr double BloomFilterMinFPR = 0.0001;

/**
 * @brief Highest accepted false-positive rate for a bloom filter.
 */
constexpr double BloomFilterMaxFPR = 0.05;

/**
 * @brief Recommended false-positive rate when a caller has no specific target.
 *
 * Sizing follows the Arrow formula, so a body holds roughly 0.72 members per byte at this
 * rate: a 64 MiB body (the default proxy.maxBloomFilterSize) holds about 48.6M members.
 * Bodies are powers of two, so a member count just past a tier boundary doubles the blob and
 * raising the rate slightly is usually the cheaper fix.
 */
constexpr double BloomFilterDefaultFPR = 0.005;

/**
 * @brief Builds the blob a bloom_match(field, {blob}) filter expression consumes.
 *
 * Building the filter on the client and shipping the compact blob lets large membership sets
 * pass the proxy gRPC receive limit, which a raw value list would exceed for the same set.
 * The proxy embeds the blob verbatim after validating the envelope and never rebuilds the
 * filter, so the bytes are a wire contract. They match the Go and Python SDKs, which have
 * shipped; the Java, Node and Rust ports target the same format and are still in flight.
 *
 * Prefer BloomFilterTemplate() unless the membership set is too large to materialise as a
 * vector. The filter is a fixed-size bit array sized from @a n and the false-positive rate
 * alone -- never from the values -- and insertion is order-independent and idempotent, so
 * members can be streamed in from a cursor or a file and dropped as they go:
 *
 * @code
 * milvus::BloomFilterBuilder builder(10000000, milvus::BloomFilterDefaultFPR);
 * while (cursor.Next()) {
 *     builder.AddInt64(cursor.Id());
 * }
 * arguments.AddFilterTemplate("bf", builder.BuildTemplate());
 * arguments.SetFilter("bloom_match(user_id, {bf})");
 * @endcode
 *
 * Build the filter from the SAME value domain as the target field: integer fields hash int64,
 * VARCHAR fields hash UTF-8. The envelope records which domains were inserted, so a
 * wrong-domain blob is rejected by the proxy with a parameter error rather than silently
 * matching nothing. A builder may record both domains, which suits a JSON path that
 * legitimately holds integers and strings alike.
 */
class MILVUS_SDK_API BloomFilterBuilder {
 public:
    /**
     * @brief Constructor.
     *
     * @param n expected number of distinct members. It only drives sizing, so inserting more
     *          is allowed and simply raises the effective false-positive rate above the
     *          target; a rough estimate is enough.
     * @param fpr false-positive rate, in [BloomFilterMinFPR, BloomFilterMaxFPR].
     *
     * Note: this constructor throws std::runtime_error when @a fpr is NaN or out of range.
     * Use EstimateBloomFilterSize() first if the rate comes from untrusted input.
     */
    // Keep implicit construction for source compatibility with existing SDK users.
    // NOLINTNEXTLINE(google-explicit-constructor)
    BloomFilterBuilder(uint64_t n, double fpr = BloomFilterDefaultFPR);

    /**
     * @brief Insert an integer member, hashed as its 8-byte little-endian encoding.
     */
    BloomFilterBuilder&
    AddInt64(int64_t value);

    /**
     * @brief Insert a string member, hashed as its raw UTF-8 bytes.
     */
    BloomFilterBuilder&
    AddString(const std::string& value);

    /**
     * @brief Insert a whole vector of integer members.
     */
    BloomFilterBuilder&
    AddInt64s(const std::vector<int64_t>& values);

    /**
     * @brief Insert a whole vector of string members.
     */
    BloomFilterBuilder&
    AddStrings(const std::vector<std::string>& values);

    /**
     * @brief The value domains recorded so far. Zero means nothing was inserted, and such a
     * filter matches no row.
     */
    uint8_t
    Domains() const;

    /**
     * @brief The number of 32-byte blocks in the filter body.
     */
    uint32_t
    NumBlocks() const;

    /**
     * @brief Stamp the envelope header and return the blob.
     */
    std::vector<uint8_t>
    Build() const;

    /**
     * @brief Same blob as Build(), wrapped as a JSON binary value so it can be handed to
     * QueryArguments::AddFilterTemplate() / SearchArguments::AddFilterTemplate() directly.
     *
     * Binary rather than a JSON string: the filter body is not valid UTF-8, and proto3 bytes
     * has no UTF-8 constraint, so the blob travels raw with no base64 inflation.
     */
    nlohmann::json
    BuildTemplate() const;

 private:
    void
    addHash(uint64_t hash);

    std::vector<uint8_t> buf_;
    uint32_t num_blocks_{0};
    uint64_t n_declared_{0};
    double fpr_{BloomFilterDefaultFPR};
    uint8_t domains_{0};
};

/**
 * @brief Build a filter over an integer membership set, ready for AddFilterTemplate().
 *
 * @param members the membership set
 * @param fpr false-positive rate, in [BloomFilterMinFPR, BloomFilterMaxFPR]
 * @param output receives the JSON binary value wrapping the blob
 */
MILVUS_SDK_API Status
BloomFilterTemplate(const std::vector<int64_t>& members, double fpr, nlohmann::json& output);

/**
 * @brief Build a filter over a string membership set, ready for AddFilterTemplate().
 */
MILVUS_SDK_API Status
BloomFilterTemplate(const std::vector<std::string>& members, double fpr, nlohmann::json& output);

/**
 * @brief The exact byte length a filter for @a n members at @a fpr will occupy, without
 * allocating it or hashing anything.
 *
 * Use it to check a planned filter against the proxy limits before building it: the body must
 * fit proxy.maxBloomFilterSize (64 MiB by default) and the whole request must fit
 * proxy.grpc.serverMaxRecvSize (128 MiB by default).
 */
MILVUS_SDK_API Status
EstimateBloomFilterSize(uint64_t n, double fpr, uint64_t& output);

}  // namespace milvus
