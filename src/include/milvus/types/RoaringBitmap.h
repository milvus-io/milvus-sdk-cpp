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
#include <type_traits>
#include <vector>

#include "milvus/Export.h"
#include "milvus/Status.h"

namespace milvus {

/**
 * @brief Highest number of high containers (distinct value >> 32 groups) a blob may carry.
 */
constexpr uint64_t RoaringBitmapMaxHighContainers = 1ULL << 18;

/**
 * @brief Highest estimated decoded size, in bytes, the server accepts for a blob.
 *
 * The estimate is body_length + 128 * high_container_count + 64 * low_container_count, which
 * bounds what the decoded bitmap costs the query node rather than what the blob costs the wire.
 */
constexpr uint64_t RoaringBitmapMaxDecodedSize = 64ULL * 1024 * 1024;

/**
 * @brief Highest body length, in bytes, the server accepts for a blob.
 */
constexpr uint64_t RoaringBitmapMaxBodySize = 128ULL * 1024 * 1024;

/**
 * @brief What a built blob will contain, without building it.
 *
 * The counts are what the limits are enforced against, so a caller sizing a membership set
 * against RoaringBitmapMaxDecodedSize can read them off the builder instead of building a blob
 * the server would reject.
 */
struct RoaringBitmapStats {
    /** @brief number of distinct members */
    uint64_t cardinality{0};
    /** @brief number of distinct value >> 32 groups */
    uint64_t high_container_count{0};
    /** @brief number of 16-bit containers across all groups */
    uint64_t low_container_count{0};
    /** @brief containers serialized as a sorted uint16 array */
    uint64_t array_containers{0};
    /** @brief containers serialized as a 8192-byte bitmap */
    uint64_t bitmap_containers{0};
    /** @brief containers serialized as a run list */
    uint64_t run_containers{0};
    /** @brief length of the body, i.e. of the blob minus its 32-byte header */
    uint64_t body_length{0};
    /** @brief body_length + 128 * high_container_count + 64 * low_container_count */
    uint64_t estimated_decoded_size{0};
};

/**
 * @brief Builds the blob a roaring_match(field, {blob}) filter expression consumes.
 *
 * roaring_match tests exact membership of an integer field in a client-supplied set. Shipping
 * the set as a compact roaring bitmap rather than as a literal list lets a membership test over
 * millions of ids pass the proxy gRPC receive limit, which the equivalent `in [...]` list would
 * exceed. The proxy embeds the blob verbatim after validating the envelope and never rebuilds
 * it, so the bytes are a wire contract: they are identical to what the Go SDK
 * (client/roaringfilter) produces for the same members.
 *
 * See docs/design-docs/design_docs/20260714-roaring-exact-membership-expression.md in the
 * milvus repo, and milvus-io/milvus#51968 for the server side.
 *
 * The builder is the reusable half of the feature: building a large bitmap is the expensive
 * part, so build it once and hand the same value to every query that needs it.
 *
 * @code
 * milvus::RoaringBitmapBuilder builder;
 * while (cursor.Next()) {
 *     builder.AddInt64(cursor.Id());
 * }
 * const auto ids = builder.BuildTemplate();   // build once
 * for (...) {
 *     arguments.AddFilterTemplate("ids", ids);  // reuse across queries
 *     arguments.SetFilter("roaring_match(user_id, {ids})");
 * }
 * @endcode
 *
 * Members are signed: a value is sign-extended to int64 and its two's complement bit pattern
 * is the uint64 bitmap key, so INT8(-1), INT32(-1) and INT64(-1) are the same member, and
 * negative values sort above positive ones inside the bitmap. That mapping is what the server
 * probes with, so a field of any signed integer width can be matched against one blob.
 *
 * Duplicate and unordered input is fine: members collapse to a distinct sorted set, and the
 * same set always yields the same bytes.
 */
class MILVUS_SDK_API RoaringBitmapBuilder {
 public:
    /**
     * @brief Insert one member.
     */
    RoaringBitmapBuilder&
    AddInt64(int64_t value);

    /**
     * @brief Insert a whole vector of members.
     */
    RoaringBitmapBuilder&
    AddInt64s(const std::vector<int64_t>& values);

    /**
     * @brief Reject floating-point members at compile time rather than truncating them.
     *
     * Without this a double argument would convert silently, so roaring_match would test
     * membership of a value the caller never asked for.
     */
    template <typename T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
    RoaringBitmapBuilder&
    AddInt64(T value) = delete;

    /**
     * @brief The number of distinct members inserted so far.
     */
    uint64_t
    Cardinality() const;

    /**
     * @brief What Build() would produce, computed without allocating the body.
     */
    RoaringBitmapStats
    Stats() const;

    /**
     * @brief Whether the current member set fits the limits the server enforces.
     *
     * Checked before the body is allocated, so an oversized set fails fast rather than after
     * materialising tens of megabytes.
     */
    Status
    Validate() const;

    /**
     * @brief Serialize the member set into the MRB1 blob.
     *
     * Note: this throws std::runtime_error when the member set exceeds a limit. Call Validate()
     * first, or use RoaringBitmapTemplate(), when the set comes from untrusted input.
     */
    std::vector<uint8_t>
    Build() const;

    /**
     * @brief Same blob as Build(), wrapped as a JSON binary value so it can be handed to
     * QueryArguments::AddFilterTemplate() / SearchArguments::AddFilterTemplate() directly.
     *
     * Binary rather than a JSON string: a roaring body is not valid UTF-8, and proto3 bytes has
     * no UTF-8 constraint, so the blob travels raw with no base64 inflation.
     */
    nlohmann::json
    BuildTemplate() const;

 private:
    void
    normalize() const;

    // The two's complement bit patterns of the members, sorted ascending as unsigned and
    // deduplicated lazily, so insertion stays O(1) per member and repeated Build() calls do not
    // re-sort.
    mutable std::vector<uint64_t> keys_;
    mutable bool normalized_{true};
};

/**
 * @brief Build a bitmap over a membership set, ready for AddFilterTemplate().
 *
 * @param members the membership set; duplicates and any order are accepted
 * @param output receives the JSON binary value wrapping the blob
 */
MILVUS_SDK_API Status
RoaringBitmapTemplate(const std::vector<int64_t>& members, nlohmann::json& output);

}  // namespace milvus
