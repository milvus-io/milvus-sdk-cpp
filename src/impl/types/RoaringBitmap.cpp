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

// The body here is the portable Roaring64 serialization -- the same format CRoaring writes
// through Roaring64Map::writeFrozen's portable path and that segcore decodes -- wrapped in the
// Milvus MRB1 envelope. See docs/design-docs/design_docs/
// 20260714-roaring-exact-membership-expression.md in the milvus repo, and the Go reference at
// client/roaringfilter.
//
// MRB1 envelope (all integers little-endian):
//
//   offset  size  field
//   0       4     magic "MRB1"
//   4       2     version      (= 1)
//   6       2     format       (1 = portable_roaring64)
//   8       8     cardinality  (number of distinct members)
//   16      8     body_length  (blob length minus 32)
//   24      8     reserved     (must be 0)
//   32      ...   body
//
// Body:
//
//   uint64  high_container_count
//   repeat, in ascending high key order:
//       uint32  high_key                          (= member >> 32)
//       <portable Roaring32 blob of the low 32 bits of that group>
//
// Portable Roaring32, for the containers of one high group in ascending 16-bit key order:
//
//   if any container is a run container:
//       uint16 12347, uint16 container_count - 1, then a bitmap of (container_count + 7) / 8
//       bytes whose bit i is set iff container i is a run container
//   else:
//       uint32 12346, uint32 container_count
//   then uint16 container_key, uint16 cardinality - 1 per container;
//   then, iff there is no run container or there are at least four containers, a uint32 offset
//   per container, relative to the start of this Roaring32 blob;
//   then the container bodies.
//
// A CRoaring reader tolerates any structurally valid encoding, so the container choices below
// only have to be legal. They are byte-identical to the Go reference anyway: identical bytes
// across the SDKs are the one cheap, total conformance signal we have, and the golden vectors
// in test/ut/testdata check exactly that.

#include "milvus/types/RoaringBitmap.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "../utils/LittleEndian.h"

namespace milvus {

namespace {

constexpr uint32_t kBitmapHeaderSize = 32;
constexpr uint16_t kSerialCookie = 12347;
constexpr uint32_t kSerialCookieNoRunContainer = 12346;
constexpr uint32_t kMaxArrayCardinality = 4096;
constexpr uint32_t kBitmapBodySize = 8192;

// The run-versus-bitmap tie-break compares against the reference implementation's IN-MEMORY
// bitmap container size -- 32 bytes of container struct plus 65536 / 8 bytes of payload on any
// 64-bit platform -- not against the 8192 bytes a bitmap container occupies on the wire. Using
// 8192 here would flip the choice for containers holding 2048..2055 runs, and the blob would
// stop matching what the Go SDK emits for the same members.
constexpr uint64_t kBitmapContainerSizeInMemory = 8224;

enum class ContainerKind { ARRAY, BITMAP, RUN };

// One 16-bit container. The values are the [begin, end) slice of the normalized key vector, so
// planning a bitmap of millions of members copies nothing.
struct ContainerPlan {
    uint16_t key{0};
    size_t begin{0};
    size_t end{0};
    uint32_t num_runs{0};
    ContainerKind kind{ContainerKind::ARRAY};
    uint32_t body_size{0};
};

struct GroupPlan {
    uint32_t key{0};
    size_t first{0};
    size_t count{0};
    bool has_run{false};
    bool has_offsets{false};
    uint32_t cookie_size{0};
    uint32_t run_bitmap_size{0};
    uint64_t size{0};
};

struct Layout {
    std::vector<GroupPlan> groups;
    std::vector<ContainerPlan> containers;
    // An empty member set still writes the 8-byte high container count, so the body is never 0.
    uint64_t body_length{8};
};

/**
 * Groups the normalized keys into high groups and 16-bit containers, picks each container's
 * encoding and sizes the whole body -- all without allocating it, so a member set that exceeds
 * a limit is rejected before tens of megabytes are materialised.
 */
Layout
PlanLayout(const std::vector<uint64_t>& keys) {
    Layout layout;
    size_t index = 0;
    while (index < keys.size()) {
        const auto high = static_cast<uint32_t>(keys[index] >> 32);
        GroupPlan group;
        group.key = high;
        group.first = layout.containers.size();
        uint64_t bodies = 0;

        while (index < keys.size() && static_cast<uint32_t>(keys[index] >> 32) == high) {
            const auto container_key = static_cast<uint16_t>(keys[index] >> 16);
            ContainerPlan container;
            container.key = container_key;
            container.begin = index;

            // Count maximal consecutive runs while walking the container's values. previous is
            // widened to uint32 so that previous + 1 cannot wrap back onto a legal value.
            uint32_t previous = 0;
            bool first = true;
            while (index < keys.size() && static_cast<uint32_t>(keys[index] >> 32) == high &&
                   static_cast<uint16_t>(keys[index] >> 16) == container_key) {
                const auto value = static_cast<uint32_t>(keys[index] & 0xFFFF);
                if (first || value != previous + 1) {
                    container.num_runs++;
                }
                previous = value;
                first = false;
                index++;
            }
            container.end = index;

            const auto cardinality = static_cast<uint32_t>(container.end - container.begin);
            const uint64_t as_run = 2 + 4ULL * container.num_runs;
            const uint64_t as_array = 2ULL * cardinality;
            if (as_run < std::min(kBitmapContainerSizeInMemory, as_array)) {
                container.kind = ContainerKind::RUN;
                container.body_size = static_cast<uint32_t>(as_run);
            } else if (cardinality <= kMaxArrayCardinality) {
                container.kind = ContainerKind::ARRAY;
                container.body_size = static_cast<uint32_t>(as_array);
            } else {
                container.kind = ContainerKind::BITMAP;
                container.body_size = kBitmapBodySize;
            }

            group.has_run = group.has_run || container.kind == ContainerKind::RUN;
            bodies += container.body_size;
            layout.containers.push_back(container);
        }

        group.count = layout.containers.size() - group.first;
        group.cookie_size = group.has_run ? 4 : 8;
        group.run_bitmap_size = group.has_run ? static_cast<uint32_t>((group.count + 7) / 8) : 0;
        // A run-bearing blob with one, two or three containers omits the offset table; a blob
        // with no run container always writes it.
        group.has_offsets = !group.has_run || group.count >= 4;
        group.size = group.cookie_size + group.run_bitmap_size + 4ULL * group.count +
                     (group.has_offsets ? 4ULL * group.count : 0) + bodies;
        layout.body_length += 4 + group.size;
        layout.groups.push_back(group);
    }
    return layout;
}

RoaringBitmapStats
StatsOf(size_t cardinality, const Layout& layout) {
    RoaringBitmapStats stats;
    stats.cardinality = cardinality;
    stats.high_container_count = layout.groups.size();
    stats.low_container_count = layout.containers.size();
    for (const auto& container : layout.containers) {
        switch (container.kind) {
            case ContainerKind::ARRAY:
                stats.array_containers++;
                break;
            case ContainerKind::BITMAP:
                stats.bitmap_containers++;
                break;
            case ContainerKind::RUN:
                stats.run_containers++;
                break;
        }
    }
    stats.body_length = layout.body_length;
    stats.estimated_decoded_size =
        layout.body_length + 128 * stats.high_container_count + 64 * stats.low_container_count;
    return stats;
}

// The two container counts, taken in one pass over the sorted keys without allocating anything.
struct BucketCounts {
    uint64_t high_container_count{0};
    uint64_t low_container_count{0};
};

BucketCounts
CountBuckets(const std::vector<uint64_t>& keys) {
    BucketCounts counts;
    if (keys.empty()) {
        return counts;
    }
    counts.high_container_count = 1;
    counts.low_container_count = 1;
    for (size_t i = 1; i < keys.size(); i++) {
        if ((keys[i] >> 32) != (keys[i - 1] >> 32)) {
            counts.high_container_count++;
        }
        if ((keys[i] >> 16) != (keys[i - 1] >> 16)) {
            counts.low_container_count++;
        }
    }
    return counts;
}

Status
CheckHighContainerLimit(uint64_t high_container_count) {
    if (high_container_count > RoaringBitmapMaxHighContainers) {
        return {StatusCode::INVALID_ARGUMENT, "Roaring bitmap high-container count " +
                                                  std::to_string(high_container_count) + " exceeds maximum " +
                                                  std::to_string(RoaringBitmapMaxHighContainers)};
    }
    return Status::OK();
}

// Both limits, decided from the counts alone, before PlanLayout() allocates a plan per container.
//
// The high-container count is exact, so that limit is compared as-is. The decoded-size estimate is
// not yet exact -- it also includes the body length, which does not exist until the layout does --
// but the per-container overhead alone is a lower bound on it, so a set whose overhead already
// exceeds the cap cannot fit however small its body turns out to be. Rejecting on that is sound.
//
// The message says the size is "at least" the overhead rather than quoting the overhead as the
// size. The distinction matters: a caller reading this error has to know what to shrink, and an
// understated figure would send them back with a set that still does not fit. Borderline sets,
// where the body is what tips the estimate over, fall through to CheckLimits() and get the exact
// number there.
Status
CheckBucketLimits(const BucketCounts& counts) {
    auto high_container_status = CheckHighContainerLimit(counts.high_container_count);
    if (!high_container_status.IsOk()) {
        return high_container_status;
    }
    const uint64_t overhead = counts.high_container_count * 128 + counts.low_container_count * 64;
    if (overhead > RoaringBitmapMaxDecodedSize) {
        return {StatusCode::INVALID_ARGUMENT, "Roaring bitmap estimated decoded size is at least " +
                                                  std::to_string(overhead) + ", exceeding maximum " +
                                                  std::to_string(RoaringBitmapMaxDecodedSize)};
    }
    return Status::OK();
}

Status
CheckLimits(const RoaringBitmapStats& stats) {
    auto high_container_status = CheckHighContainerLimit(stats.high_container_count);
    if (!high_container_status.IsOk()) {
        return high_container_status;
    }
    if (stats.estimated_decoded_size > RoaringBitmapMaxDecodedSize) {
        return {StatusCode::INVALID_ARGUMENT, "Roaring bitmap estimated decoded size " +
                                                  std::to_string(stats.estimated_decoded_size) + " exceeds maximum " +
                                                  std::to_string(RoaringBitmapMaxDecodedSize)};
    }
    // Unreachable while the decoded-size limit holds, since that estimate is the body length
    // plus a non-negative per-container term. The server checks it independently, so it is
    // mirrored here rather than assumed away.
    if (stats.body_length > RoaringBitmapMaxBodySize) {
        return {StatusCode::INVALID_ARGUMENT, "Roaring bitmap body too large: body length " +
                                                  std::to_string(stats.body_length) + " exceeds maximum " +
                                                  std::to_string(RoaringBitmapMaxBodySize)};
    }
    return Status::OK();
}

void
WriteBody(uint8_t* body, const std::vector<uint64_t>& keys, const Layout& layout) {
    WriteU64LE(body, layout.groups.size());
    size_t pos = 8;

    for (const auto& group : layout.groups) {
        WriteU32LE(body + pos, group.key);
        pos += 4;

        // Everything from here to the end of this group is one portable Roaring32 blob, and the
        // offsets below are relative to this point.
        if (group.has_run) {
            WriteU16LE(body + pos, kSerialCookie);
            WriteU16LE(body + pos + 2, static_cast<uint16_t>(group.count - 1));
            pos += 4;
            // The run bitmap, and its trailing pad bits, start out zero.
            for (size_t i = 0; i < group.count; i++) {
                if (layout.containers[group.first + i].kind == ContainerKind::RUN) {
                    body[pos + i / 8] |= static_cast<uint8_t>(1u << (i % 8));
                }
            }
            pos += group.run_bitmap_size;
        } else {
            WriteU32LE(body + pos, kSerialCookieNoRunContainer);
            WriteU32LE(body + pos + 4, static_cast<uint32_t>(group.count));
            pos += 8;
        }

        for (size_t i = 0; i < group.count; i++) {
            const auto& container = layout.containers[group.first + i];
            WriteU16LE(body + pos, container.key);
            // Minus one: a container holding all 65536 values still has to fit a uint16.
            WriteU16LE(body + pos + 2, static_cast<uint16_t>(container.end - container.begin - 1));
            pos += 4;
        }

        if (group.has_offsets) {
            uint32_t offset = group.cookie_size + group.run_bitmap_size + static_cast<uint32_t>(8 * group.count);
            for (size_t i = 0; i < group.count; i++) {
                WriteU32LE(body + pos, offset);
                offset += layout.containers[group.first + i].body_size;
                pos += 4;
            }
        }

        for (size_t i = 0; i < group.count; i++) {
            const auto& container = layout.containers[group.first + i];
            switch (container.kind) {
                case ContainerKind::ARRAY: {
                    for (size_t j = container.begin; j < container.end; j++) {
                        WriteU16LE(body + pos, static_cast<uint16_t>(keys[j]));
                        pos += 2;
                    }
                    break;
                }
                case ContainerKind::BITMAP: {
                    // Word v >> 6, bit v & 63, stored little-endian -- which is byte v >> 3,
                    // bit v & 7. The 8192 bytes start out zero.
                    for (size_t j = container.begin; j < container.end; j++) {
                        const auto value = static_cast<uint16_t>(keys[j]);
                        body[pos + (value >> 3)] |= static_cast<uint8_t>(1u << (value & 7));
                    }
                    pos += kBitmapBodySize;
                    break;
                }
                case ContainerKind::RUN: {
                    WriteU16LE(body + pos, static_cast<uint16_t>(container.num_runs));
                    pos += 2;
                    size_t j = container.begin;
                    while (j < container.end) {
                        const auto start = static_cast<uint32_t>(keys[j] & 0xFFFF);
                        uint32_t last = start;
                        j++;
                        while (j < container.end && static_cast<uint32_t>(keys[j] & 0xFFFF) == last + 1) {
                            last++;
                            j++;
                        }
                        WriteU16LE(body + pos, static_cast<uint16_t>(start));
                        // Minus one, so a run covering the whole container still fits a uint16.
                        WriteU16LE(body + pos + 2, static_cast<uint16_t>(last - start));
                        pos += 4;
                    }
                    break;
                }
            }
        }
    }
}

}  // namespace

RoaringBitmapBuilder&
RoaringBitmapBuilder::AddInt64(int64_t value) {
    // Sign-extend, then reinterpret the two's complement bit pattern as the bitmap key. This is
    // the mapping the server probes with, so INT8(-1) and INT64(-1) are the same member and
    // negative values land in the top half of the key space.
    keys_.push_back(static_cast<uint64_t>(value));
    normalized_ = false;
    return *this;
}

RoaringBitmapBuilder&
RoaringBitmapBuilder::AddInt64s(const std::vector<int64_t>& values) {
    if (values.empty()) {
        return *this;
    }
    keys_.reserve(keys_.size() + values.size());
    for (const auto value : values) {
        keys_.push_back(static_cast<uint64_t>(value));
    }
    normalized_ = false;
    return *this;
}

void
RoaringBitmapBuilder::normalize() const {
    if (normalized_) {
        return;
    }
    // Unsigned order, which is what the bitmap layout is defined on: -1 sorts above 5, not
    // below it. Sorting the signed values instead would group and order the containers wrongly
    // for any member set that straddles zero.
    std::sort(keys_.begin(), keys_.end());
    keys_.erase(std::unique(keys_.begin(), keys_.end()), keys_.end());
    normalized_ = true;
}

uint64_t
RoaringBitmapBuilder::Cardinality() const {
    normalize();
    return keys_.size();
}

RoaringBitmapStats
RoaringBitmapBuilder::Stats() const {
    normalize();
    return StatsOf(keys_.size(), PlanLayout(keys_));
}

Status
RoaringBitmapBuilder::Validate() const {
    normalize();
    // Both limits are decided from the counting pass first. PlanLayout() would otherwise spend
    // tens of bytes per container laying out a set that was never going to be accepted -- either
    // a pathologically sparse one, where shuffled full-range int64 ids land in nearly one high
    // container each, or a wide one, where the per-container overhead alone already exceeds the
    // decoded-size cap.
    auto status = CheckBucketLimits(CountBuckets(keys_));
    if (!status.IsOk()) {
        return status;
    }
    return CheckLimits(Stats());
}

std::vector<uint8_t>
RoaringBitmapBuilder::Build() const {
    normalize();
    const auto bucket_status = CheckBucketLimits(CountBuckets(keys_));
    if (!bucket_status.IsOk()) {
        throw std::runtime_error(bucket_status.Message());
    }

    const auto layout = PlanLayout(keys_);
    const auto stats = StatsOf(keys_.size(), layout);
    const auto status = CheckLimits(stats);
    if (!status.IsOk()) {
        throw std::runtime_error(status.Message());
    }

    std::vector<uint8_t> blob(static_cast<size_t>(kBitmapHeaderSize + layout.body_length), 0);
    std::memcpy(blob.data(), "MRB1", 4);
    WriteU16LE(blob.data() + 4, 1);  // version
    WriteU16LE(blob.data() + 6, 1);  // format = portable_roaring64
    WriteU64LE(blob.data() + 8, stats.cardinality);
    WriteU64LE(blob.data() + 16, layout.body_length);
    // blob[24..31] stays zero (reserved).
    WriteBody(blob.data() + kBitmapHeaderSize, keys_, layout);
    return blob;
}

nlohmann::json
RoaringBitmapBuilder::BuildTemplate() const {
    return nlohmann::json::binary(Build());
}

Status
RoaringBitmapTemplate(const std::vector<int64_t>& members, nlohmann::json& output) {
    RoaringBitmapBuilder builder;
    builder.AddInt64s(members);
    auto status = builder.Validate();
    if (!status.IsOk()) {
        return status;
    }
    output = builder.BuildTemplate();
    return Status::OK();
}

}  // namespace milvus
