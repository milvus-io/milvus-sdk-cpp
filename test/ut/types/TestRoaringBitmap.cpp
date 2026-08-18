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

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <milvus/thirdparty/nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "milvus/request/dml/DeleteRequest.h"
#include "milvus/types/QueryArguments.h"
#include "milvus/types/RoaringBitmap.h"
#include "milvus/types/SearchArguments.h"
#include "utils/DqlUtils.h"

namespace {

// The blob is the whole contract: the SDKs interoperate only if they emit the same bytes for
// the same members, because the proxy embeds the blob verbatim and never rebuilds the bitmap.
// These are the fixtures the Go reference (client/roaringfilter) is checked against; every one
// of them also passes the server's own validator and decodes under CRoaring, the library
// segcore probes with.
//
// They live as a data file rather than inlined in this source the way the bloom vectors are:
// the set is 90 KB, the other SDK repos carry the identical file, and MSVC caps a single string
// literal at 16380 bytes. MILVUS_UT_TESTDATA_DIR is an absolute path baked in by CMake, so the
// binary finds them from any working directory.
const char* kGoldenVectorsPath = MILVUS_UT_TESTDATA_DIR "/roaring_golden_vectors.json";

nlohmann::json
LoadGoldenVectors() {
    std::ifstream file(kGoldenVectorsPath);
    EXPECT_TRUE(file.is_open()) << "cannot open golden vectors at " << kGoldenVectorsPath;
    std::stringstream text;
    text << file.rdbuf();
    return nlohmann::json::parse(text.str());
}

std::vector<uint8_t>
Base64Decode(const std::string& text) {
    static const std::string kAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<uint8_t> out;
    uint32_t buffer = 0;
    int bits = 0;
    for (char one : text) {
        const auto position = kAlphabet.find(one);
        if (position == std::string::npos) {
            continue;  // padding and whitespace
        }
        buffer = (buffer << 6) | static_cast<uint32_t>(position);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((buffer >> bits) & 0xFF));
        }
    }
    return out;
}

// A fixture lists its members as {start, count, step} ranges, and the values are decimal strings
// because an int64 does not survive a JSON number in every language that reads this file.
std::vector<int64_t>
ExpandMembers(const nlohmann::json& members) {
    std::vector<int64_t> out;
    for (const auto& one : members) {
        const auto start = static_cast<uint64_t>(std::stoll(one["start"].get<std::string>()));
        const auto step = static_cast<uint64_t>(std::stoll(one["step"].get<std::string>()));
        const auto count = one["count"].get<uint64_t>();
        for (uint64_t i = 0; i < count; i++) {
            out.push_back(static_cast<int64_t>(start + step * i));
        }
    }
    return out;
}

uint64_t
KindCount(const nlohmann::json& kinds, const char* kind) {
    return kinds.contains(kind) ? kinds[kind].get<uint64_t>() : 0;
}

// Reports the first differing offset, which is what pinpoints the rule that is wrong.
std::string
Difference(const std::vector<uint8_t>& actual, const std::vector<uint8_t>& expected) {
    size_t at = 0;
    while (at < actual.size() && at < expected.size() && actual[at] == expected[at]) {
        at++;
    }
    std::stringstream out;
    out << "size " << actual.size() << " want " << expected.size() << ", first difference at offset " << at;
    if (at < actual.size() && at < expected.size()) {
        out << " (got " << static_cast<int>(actual[at]) << " want " << static_cast<int>(expected[at]) << ")";
    }
    return out.str();
}

uint16_t
ReadU16LE(const std::vector<uint8_t>& blob, size_t offset) {
    return static_cast<uint16_t>(blob[offset] | blob[offset + 1] << 8);
}

uint32_t
ReadU32LE(const std::vector<uint8_t>& blob, size_t offset) {
    return static_cast<uint32_t>(blob[offset]) | static_cast<uint32_t>(blob[offset + 1]) << 8 |
           static_cast<uint32_t>(blob[offset + 2]) << 16 | static_cast<uint32_t>(blob[offset + 3]) << 24;
}

uint64_t
ReadU64LE(const std::vector<uint8_t>& blob, size_t offset) {
    return static_cast<uint64_t>(ReadU32LE(blob, offset)) | static_cast<uint64_t>(ReadU32LE(blob, offset + 4)) << 32;
}

std::vector<uint8_t>
BuildOne(int64_t member) {
    milvus::RoaringBitmapBuilder builder;
    return builder.AddInt64(member).Build();
}

// Detects whether AddInt64 accepts a given argument type. The floating-point overload is
// deleted, so overload resolution fails in the immediate context and this reports false.
template <typename T, typename = void>
struct AcceptsMember : std::false_type {};

template <typename T>
struct AcceptsMember<T, decltype(void(std::declval<milvus::RoaringBitmapBuilder&>().AddInt64(std::declval<T>())))>
    : std::true_type {};

}  // namespace

TEST(RoaringBitmapTest, MatchesSharedGoldenVectors) {
    auto fixture = LoadGoldenVectors();
    ASSERT_EQ("MRB1", fixture["spec"].get<std::string>());
    ASSERT_EQ(29, fixture["cases"].size());

    for (const auto& one : fixture["cases"]) {
        const auto name = one["name"].get<std::string>();
        const auto expected = Base64Decode(one["blob_base64"].get<std::string>());

        milvus::RoaringBitmapBuilder builder;
        builder.AddInt64s(ExpandMembers(one["members"]));
        const auto blob = builder.Build();
        EXPECT_EQ(expected, blob) << "golden vector " << name << ": " << Difference(blob, expected);

        // The counts the limits are enforced against have to agree with the fixture too, since
        // a caller sizes a membership set from them without building anything.
        const auto stats = builder.Stats();
        EXPECT_EQ(one["cardinality"].get<uint64_t>(), stats.cardinality) << name;
        EXPECT_EQ(one["body_length"].get<uint64_t>(), stats.body_length) << name;
        EXPECT_EQ(one["high_container_count"].get<uint64_t>(), stats.high_container_count) << name;
        EXPECT_EQ(one["low_container_count"].get<uint64_t>(), stats.low_container_count) << name;
        const auto& kinds = one["container_kinds"];
        EXPECT_EQ(KindCount(kinds, "array"), stats.array_containers) << name;
        EXPECT_EQ(KindCount(kinds, "bitmap"), stats.bitmap_containers) << name;
        EXPECT_EQ(KindCount(kinds, "run"), stats.run_containers) << name;
    }
}

TEST(RoaringBitmapTest, LaysOutTheEnvelopeHeader) {
    milvus::RoaringBitmapBuilder builder;
    builder.AddInt64s({1, 2, 3, 1});
    const auto blob = builder.Build();

    EXPECT_EQ("MRB1", std::string(reinterpret_cast<const char*>(blob.data()), 4));
    EXPECT_EQ(1, ReadU16LE(blob, 4));   // version
    EXPECT_EQ(1, ReadU16LE(blob, 6));   // format: portable_roaring64
    EXPECT_EQ(3u, ReadU64LE(blob, 8));  // cardinality counts distinct members
    EXPECT_EQ(blob.size() - 32, ReadU64LE(blob, 16));
    // Reserved bytes must be zero; the server rejects a blob that sets them.
    EXPECT_EQ(0u, ReadU64LE(blob, 24));
    EXPECT_EQ(1u, ReadU64LE(blob, 32));  // one high container
}

// Sign-extend to int64, then reinterpret the two's complement bit pattern as the bitmap key.
// Zero-extending a narrow value, ZigZag or a 2^63 bias would all put the member somewhere else,
// and the server would then find nothing where it probes.
TEST(RoaringBitmapTest, MapsSignedMembersToUnsignedKeys) {
    const std::pair<int64_t, uint64_t> cases[] = {
        {-1, 0xffffffffffffffffULL},
        {-128, 0xffffffffffffff80ULL},
        {-32768, 0xffffffffffff8000ULL},
        {-2147483648LL, 0xffffffff80000000ULL},
        {-9223372036854775807LL - 1, 0x8000000000000000ULL},
        {0, 0x0000000000000000ULL},
        {42, 0x000000000000002aULL},
        {9223372036854775807LL, 0x7fffffffffffffffULL},
    };

    for (const auto& one : cases) {
        const auto blob = BuildOne(one.first);
        // A single member is one array container, so the layout is fixed: high key at 40, the
        // container key in the descriptive header at 52, the value itself at 60.
        ASSERT_EQ(62u, blob.size()) << one.first;
        EXPECT_EQ(static_cast<uint32_t>(one.second >> 32), ReadU32LE(blob, 40)) << one.first;
        EXPECT_EQ(static_cast<uint16_t>(one.second >> 16), ReadU16LE(blob, 52)) << one.first;
        EXPECT_EQ(static_cast<uint16_t>(one.second), ReadU16LE(blob, 60)) << one.first;
    }

    // A narrow signed member is the same member as its int64 widening, so one blob serves an
    // INT8 field and an INT64 field holding the same value.
    const auto minus_one = BuildOne(-1);
    EXPECT_EQ(minus_one, BuildOne(static_cast<int8_t>(-1)));
    EXPECT_EQ(minus_one, BuildOne(static_cast<int16_t>(-1)));
    EXPECT_EQ(minus_one, BuildOne(static_cast<int32_t>(-1)));
    // ... and INT8(-1) is emphatically not the member 255.
    EXPECT_NE(minus_one, BuildOne(255));
}

TEST(RoaringBitmapTest, OrdersContainersOnTheUnsignedKey) {
    milvus::RoaringBitmapBuilder builder;
    builder.AddInt64s({-1, 5});
    const auto blob = builder.Build();

    // Two high containers: 5 lives under high key 0 and -1 under 0xffffffff. Ordering on the
    // signed value would emit them the other way round, and the server rejects a body whose
    // high keys do not ascend.
    ASSERT_EQ(2u, ReadU64LE(blob, 32));
    EXPECT_EQ(0x00000000u, ReadU32LE(blob, 40));
    const auto second_group = 40 + 4 + (8 + 4 + 4 + 2);  // high key + cookie + descriptive + offset + value
    EXPECT_EQ(0xffffffffu, ReadU32LE(blob, second_group));
}

TEST(RoaringBitmapTest, CollapsesDuplicatesAndIgnoresInputOrder) {
    milvus::RoaringBitmapBuilder ordered;
    ordered.AddInt64s({-9223372036854775807LL - 1, -1, 0, 1, 42, 9223372036854775807LL});

    milvus::RoaringBitmapBuilder shuffled;
    shuffled.AddInt64s({42, 9223372036854775807LL, -1, 0, 1, -9223372036854775807LL - 1});

    milvus::RoaringBitmapBuilder repeated;
    repeated.AddInt64(42).AddInt64(-1).AddInt64(42).AddInt64s({0, 1, -1, 42});
    repeated.AddInt64s({9223372036854775807LL, -9223372036854775807LL - 1, 0});

    EXPECT_EQ(6u, ordered.Cardinality());
    EXPECT_EQ(ordered.Build(), shuffled.Build());
    EXPECT_EQ(ordered.Build(), repeated.Build());
    EXPECT_EQ(6u, repeated.Cardinality());
}

TEST(RoaringBitmapTest, BuildsAnEmptyBitmap) {
    milvus::RoaringBitmapBuilder builder;
    const auto blob = builder.Build();

    // No members still means a well-formed blob: the body is the 8-byte high container count
    // and nothing else, so body_length is 8 rather than 0.
    EXPECT_EQ(40u, blob.size());
    EXPECT_EQ(0u, ReadU64LE(blob, 8));
    EXPECT_EQ(8u, ReadU64LE(blob, 16));
    EXPECT_EQ(0u, ReadU64LE(blob, 32));
    EXPECT_EQ(0u, builder.Cardinality());
    EXPECT_TRUE(builder.Validate().IsOk());
}

TEST(RoaringBitmapTest, RejectsTooManyHighContainers) {
    std::vector<int64_t> members;
    members.reserve(milvus::RoaringBitmapMaxHighContainers + 1);
    for (int64_t i = 0; i <= static_cast<int64_t>(milvus::RoaringBitmapMaxHighContainers); i++) {
        members.push_back(i << 32);
    }

    milvus::RoaringBitmapBuilder at_limit;
    at_limit.AddInt64s(std::vector<int64_t>(members.begin(), members.end() - 1));
    EXPECT_EQ(milvus::RoaringBitmapMaxHighContainers, at_limit.Stats().high_container_count);
    EXPECT_TRUE(at_limit.Validate().IsOk()) << at_limit.Validate().Message();

    milvus::RoaringBitmapBuilder over_limit;
    over_limit.AddInt64s(members);
    const auto status = over_limit.Validate();
    EXPECT_FALSE(status.IsOk());
    EXPECT_NE(std::string::npos, status.Message().find("high-container count 262145")) << status.Message();
    EXPECT_NE(std::string::npos, status.Message().find("262144")) << status.Message();
    // The limits are enforced by Build() too, so a caller that skips Validate() cannot ship a
    // blob the proxy would reject.
    EXPECT_THROW(over_limit.Build(), std::runtime_error);
}

TEST(RoaringBitmapTest, RejectsAnOversizedDecodedBitmap) {
    // One member per 16-bit container is the cheapest way to a large decoded bitmap: the body
    // stays small while the per-container term dominates the estimate. Thirteen high containers
    // fit, fourteen do not.
    auto members_for = [](int64_t groups) {
        std::vector<int64_t> members;
        members.reserve(static_cast<size_t>(groups) * 65536);
        for (int64_t group = 0; group < groups; group++) {
            for (int64_t container = 0; container < 65536; container++) {
                members.push_back((group << 32) | (container << 16));
            }
        }
        return members;
    };

    milvus::RoaringBitmapBuilder under;
    under.AddInt64s(members_for(13));
    EXPECT_LE(under.Stats().estimated_decoded_size, milvus::RoaringBitmapMaxDecodedSize);
    EXPECT_TRUE(under.Validate().IsOk()) << under.Validate().Message();

    milvus::RoaringBitmapBuilder over;
    over.AddInt64s(members_for(14));
    const auto stats = over.Stats();
    EXPECT_GT(stats.estimated_decoded_size, milvus::RoaringBitmapMaxDecodedSize);
    // The estimate is what the server checks, and it is far larger than the 9 MB body: a blob
    // that fits the wire limit can still decode into more than the query node will accept.
    EXPECT_LT(stats.body_length, milvus::RoaringBitmapMaxDecodedSize);

    const auto status = over.Validate();
    EXPECT_FALSE(status.IsOk());
    EXPECT_NE(std::string::npos, status.Message().find("estimated decoded size")) << status.Message();
    EXPECT_NE(std::string::npos, status.Message().find("67108864")) << status.Message();

    nlohmann::json value;
    EXPECT_FALSE(milvus::RoaringBitmapTemplate(members_for(14), value).IsOk());
}

// C++ types the members for us, so the input-type rejection the other SDKs do at run time is a
// compile-time matter here: a string or a null cannot reach AddInt64 at all, and every value in
// [-2^63, 2^63) is a legal member. The type system leaves two holes, and both overloads are
// deleted: a float would convert silently and shift the member, and a uint64_t above INT64_MAX
// would wrap, so UINT64_MAX would silently become the member -1.
TEST(RoaringBitmapTest, RejectsFloatingPointMembers) {
    static_assert(AcceptsMember<int64_t>::value, "int64 members must be accepted");
    static_assert(AcceptsMember<int8_t>::value, "narrow signed members must be accepted");
    static_assert(AcceptsMember<int32_t>::value, "narrow signed members must be accepted");
    static_assert(!AcceptsMember<double>::value, "a double member must not compile");
    static_assert(!AcceptsMember<float>::value, "a float member must not compile");
    static_assert(!AcceptsMember<const char*>::value, "a string member must not compile");

    // Unsigned types that widen losslessly stay usable; only the 64-bit ones, which are the only
    // ones that can carry a value int64_t cannot represent, are rejected.
    static_assert(AcceptsMember<uint8_t>::value, "narrow unsigned members must be accepted");
    static_assert(AcceptsMember<uint16_t>::value, "narrow unsigned members must be accepted");
    static_assert(AcceptsMember<uint32_t>::value, "narrow unsigned members must be accepted");
    static_assert(!AcceptsMember<uint64_t>::value, "a uint64 member must not compile");
    static_assert(!AcceptsMember<unsigned long long>::value,  // NOLINT(runtime/int)
                  "a uint64 member must not compile");

    // The whole int64 range is legal, including both ends.
    milvus::RoaringBitmapBuilder builder;
    builder.AddInt64s({-9223372036854775807LL - 1, 9223372036854775807LL});
    EXPECT_EQ(2u, builder.Cardinality());
}

TEST(RoaringBitmapTest, BuildsATemplateReadyBinaryValue) {
    nlohmann::json value;
    const auto status = milvus::RoaringBitmapTemplate({1, 2, 3, -1}, value);
    ASSERT_TRUE(status.IsOk()) << status.Message();

    // Binary, not a string: a roaring body is not valid UTF-8, so a string round trip would
    // corrupt it as well as inflating it through base64.
    ASSERT_TRUE(value.is_binary());

    milvus::RoaringBitmapBuilder expected;
    expected.AddInt64s({1, 2, 3, -1});
    const auto& binary = value.get_binary();
    EXPECT_EQ(expected.Build(), std::vector<uint8_t>(binary.begin(), binary.end()));
    EXPECT_EQ(expected.BuildTemplate(), value);
}

// Building a large bitmap is the expensive half of the feature, so the built value has to be
// something a caller holds and reuses rather than something a per-request call hides.
TEST(RoaringBitmapTest, IsReusableAcrossRequests) {
    milvus::RoaringBitmapBuilder builder;
    builder.AddInt64s({7, 11, 13});
    const auto ids = builder.BuildTemplate();

    // Building twice yields the same bytes, and inserting after a build keeps working.
    EXPECT_EQ(builder.Build(), builder.Build());
    EXPECT_EQ(4u, builder.AddInt64(17).Cardinality());

    milvus::QueryArguments first;
    ASSERT_TRUE(first.AddFilterTemplate("ids", ids).IsOk());
    first.SetFilter("roaring_match(user_id, {ids})");

    milvus::QueryArguments second;
    ASSERT_TRUE(second.AddFilterTemplate("ids", ids).IsOk());
    second.SetFilter("roaring_match(user_id, {ids}) and age > 30");

    EXPECT_EQ(first.FilterTemplates().at("ids"), second.FilterTemplates().at("ids"));
}

// The blob has to survive the path a caller actually uses. QueryArguments and SearchArguments
// validate through IsValidTemplate before they store anything, so a value the validator rejects
// never reaches ConvertFilterTemplates at all -- which is exactly how the bloom blob was
// accepted by the converter and refused by the setter the first time round.
TEST(RoaringBitmapTest, ReachesTheWireThroughThePublicApi) {
    nlohmann::json ids;
    ASSERT_TRUE(milvus::RoaringBitmapTemplate({-1, 0, 1, 42, 65537}, ids).IsOk());
    const auto& binary = ids.get_binary();
    const std::string expected(binary.begin(), binary.end());

    milvus::QueryArguments query;
    auto status = query.AddFilterTemplate("ids", ids);
    ASSERT_TRUE(status.IsOk()) << "QueryArguments rejected the blob: " << status.Message();

    milvus::SearchArguments search;
    status = search.AddFilterTemplate("ids", ids);
    ASSERT_TRUE(status.IsOk()) << "SearchArguments rejected the blob: " << status.Message();

    // Delete goes through the same converter, and it is the reason roaring_match exists as
    // something distinct from bloom_match: an exact filter is safe to delete through, while a
    // bloom filter's false positives would delete rows outside the intended set, so the server
    // permits roaring_match in a delete expression and rejects bloom_match.
    milvus::DeleteRequest remove;
    remove.WithFilter("roaring_match(user_id, {ids})").AddFilterTemplate("ids", nlohmann::json(ids));

    for (const auto& templates : {query.FilterTemplates(), search.FilterTemplates(), remove.FilterTemplates()}) {
        ::google::protobuf::Map<std::string, milvus::proto::schema::TemplateValue> rpc_templates;
        status = milvus::ConvertFilterTemplates(templates, &rpc_templates);
        ASSERT_TRUE(status.IsOk()) << status.Message();
        ASSERT_EQ(1, rpc_templates.count("ids"));
        // bytes_val, not string_val: an empty bytes_val here would mean the blob took the
        // string branch, where the body is not valid UTF-8 and would be corrupted as well as
        // inflated.
        EXPECT_EQ(expected, rpc_templates.at("ids").bytes_val());
    }
}
