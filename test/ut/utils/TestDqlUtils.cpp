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

#include <gmock/gmock.h>

#include "milvus/types/Constants.h"
#include "milvus/types/FieldData.h"
#include "milvus/types/FieldSchema.h"
#include "milvus/types/QueryArguments.h"
#include "milvus/types/SearchArguments.h"
#include "milvus/types/SearchResults.h"
#include "utils/Constants.h"
#include "utils/DqlUtils.h"
#include "utils/GtsDict.h"

using ::testing::ElementsAre;

class DqlUtilsTest : public ::testing::Test {};

TEST_F(DqlUtilsTest, DeduceGuaranteeTimestampTest) {
    auto ts = milvus::DeduceGuaranteeTimestamp(milvus::ConsistencyLevel::NONE, "db", "coll");
    EXPECT_EQ(ts, 1);

    ts = milvus::DeduceGuaranteeTimestamp(milvus::ConsistencyLevel::SESSION, "db", "coll");
    EXPECT_EQ(ts, 1);

    milvus::GtsDict::GetInstance().UpdateCollectionTs("db", "coll", 999);
    ts = milvus::DeduceGuaranteeTimestamp(milvus::ConsistencyLevel::NONE, "db", "coll");
    EXPECT_EQ(ts, 999);

    ts = milvus::DeduceGuaranteeTimestamp(milvus::ConsistencyLevel::SESSION, "db", "coll");
    EXPECT_EQ(ts, 999);

    ts = milvus::DeduceGuaranteeTimestamp(milvus::ConsistencyLevel::STRONG, "db", "coll");
    EXPECT_EQ(ts, milvus::GuaranteeStrongTs());

    ts = milvus::DeduceGuaranteeTimestamp(milvus::ConsistencyLevel::BOUNDED, "db", "coll");
    EXPECT_EQ(ts, 2);

    ts = milvus::DeduceGuaranteeTimestamp(milvus::ConsistencyLevel::EVENTUALLY, "db", "coll");
    EXPECT_EQ(ts, 1);
}

template <typename T, typename V>
void
TestCopyFieldData(const std::vector<V>& src_data) {
    std::string name = "dummy";
    uint64_t from = 1;
    uint64_t to = 3;
    milvus::FieldDataPtr target_field;
    auto status = milvus::CopyFieldData(nullptr, from, to, target_field);
    EXPECT_FALSE(status.IsOk());

    auto src_field = std::make_shared<T>(name, src_data);
    status = milvus::CopyFieldData(src_field, src_data.size(), 0, target_field);
    EXPECT_FALSE(status.IsOk());
    status = milvus::CopyFieldData(src_field, 0, src_data.size() + 1, target_field);
    EXPECT_TRUE(status.IsOk());

    status = milvus::CopyFieldData(src_field, from, to, target_field);
    EXPECT_TRUE(status.IsOk());

    auto target = std::dynamic_pointer_cast<T>(target_field);
    EXPECT_TRUE(target != nullptr);
    EXPECT_EQ(target->Name(), src_field->Name());
    EXPECT_EQ(target->Count(), to - from);
    for (auto i = from; i < to; i++) {
        EXPECT_EQ(target->Data().at(i - from), src_data.at(i));
    }
}

TEST_F(DqlUtilsTest, CopyFieldDataTest) {
    {
        std::vector<bool> src_data{true, false, false, true, true};
        TestCopyFieldData<milvus::BoolFieldData, bool>(src_data);
    }
    {
        std::vector<int8_t> src_data{2, 87, -23, 123, 67};
        TestCopyFieldData<milvus::Int8FieldData, int8_t>(src_data);
    }
    {
        std::vector<int16_t> src_data{234, 1234, 0, -45, 34};
        TestCopyFieldData<milvus::Int16FieldData, int16_t>(src_data);
    }
    {
        std::vector<int32_t> src_data{56756, -42, 23, 5, 2034};
        TestCopyFieldData<milvus::Int32FieldData, int32_t>(src_data);
    }
    {
        std::vector<int64_t> src_data{12234, 9999, 880, -34678, 213};
        TestCopyFieldData<milvus::Int64FieldData, int64_t>(src_data);
    }
    {
        std::vector<float> src_data{2.5, 564.12, -445.2, -9, 0};
        TestCopyFieldData<milvus::FloatFieldData, float>(src_data);
    }
    {
        std::vector<double> src_data{45, 0.0, -3.6, 5467, 43};
        TestCopyFieldData<milvus::DoubleFieldData, double>(src_data);
    }
    {
        std::vector<std::string> src_data{"hello", "world", "ok", "good", "milvus"};
        TestCopyFieldData<milvus::VarCharFieldData, std::string>(src_data);
    }
    {
        std::vector<nlohmann::json> src_data{
            nlohmann::json::parse(R"({"name":"aaa","age":18,"score":88})"), nlohmann::json::parse(R"({"flag":true})"),
            nlohmann::json::parse(R"({"name":"bbb","array":[1, 2, 3]})"),
            nlohmann::json::parse(R"({"id":10,"desc":{"flag": false}})"), nlohmann::json::parse(R"({"id":8})")};
        TestCopyFieldData<milvus::JSONFieldData, nlohmann::json>(src_data);
    }
    {
        std::vector<std::vector<bool>> src_data{{true, false}, {false, true, true}, {}, {true}, {false}};
        TestCopyFieldData<milvus::ArrayBoolFieldData, std::vector<bool>>(src_data);
    }
    {
        std::vector<std::vector<int8_t>> src_data{{2, 87}, {-23, 123}, {}, {67}, {6}};
        TestCopyFieldData<milvus::ArrayInt8FieldData, std::vector<int8_t>>(src_data);
    }
    {
        std::vector<std::vector<int16_t>> src_data{{234}, {}, {1234, 0, -45}, {34}, {}};
        TestCopyFieldData<milvus::ArrayInt16FieldData, std::vector<int16_t>>(src_data);
    }
    {
        std::vector<std::vector<int32_t>> src_data{{56756}, {-42, 23}, {}, {}, {5, 2034}};
        TestCopyFieldData<milvus::ArrayInt32FieldData, std::vector<int32_t>>(src_data);
    }
    {
        std::vector<std::vector<int64_t>> src_data{{12234, 9999}, {}, {880}, {-34678, 213}, {2}};
        TestCopyFieldData<milvus::ArrayInt64FieldData, std::vector<int64_t>>(src_data);
    }
    {
        std::vector<std::vector<float>> src_data{{2.5}, {564.12, -445.2}, {-9, 0}, {}, {2.34}};
        TestCopyFieldData<milvus::ArrayFloatFieldData, std::vector<float>>(src_data);
    }
    {
        std::vector<std::vector<double>> src_data{{}, {45, 0.0, -3.6}, {}, {5467, 43}, {}};
        TestCopyFieldData<milvus::ArrayDoubleFieldData, std::vector<double>>(src_data);
    }
    {
        std::vector<std::vector<std::string>> src_data{{}, {"hello", "world"}, {"ok"}, {"good", "milvus"}, {}};
        TestCopyFieldData<milvus::ArrayVarCharFieldData, std::vector<std::string>>(src_data);
    }
}

TEST_F(DqlUtilsTest, GetRowsFromFieldsDataTest) {
    // Two fields with 3 rows each
    auto int_field = std::make_shared<milvus::Int64FieldData>("id", std::vector<int64_t>{10, 20, 30});
    auto str_field = std::make_shared<milvus::VarCharFieldData>("name", std::vector<std::string>{"a", "b", "c"});

    std::vector<milvus::FieldDataPtr> fields{int_field, str_field};
    std::set<std::string> output_names{"id", "name"};
    milvus::EntityRows rows;
    auto status = milvus::GetRowsFromFieldsData(fields, output_names, rows);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(rows.size(), 3u);
    EXPECT_EQ(rows[0]["id"], 10);
    EXPECT_EQ(rows[0]["name"], "a");
    EXPECT_EQ(rows[2]["id"], 30);
    EXPECT_EQ(rows[2]["name"], "c");
}

TEST_F(DqlUtilsTest, GetRowsFromFieldsDataEmpty) {
    std::vector<milvus::FieldDataPtr> fields;
    std::set<std::string> output_names;
    milvus::EntityRows rows;
    auto status = milvus::GetRowsFromFieldsData(fields, output_names, rows);
    EXPECT_TRUE(status.IsOk());
    EXPECT_TRUE(rows.empty());
}

TEST_F(DqlUtilsTest, GetRowFromFieldsDataTest) {
    auto int_field = std::make_shared<milvus::Int64FieldData>("id", std::vector<int64_t>{10, 20, 30});
    auto str_field = std::make_shared<milvus::VarCharFieldData>("name", std::vector<std::string>{"a", "b", "c"});

    std::vector<milvus::FieldDataPtr> fields{int_field, str_field};
    std::set<std::string> output_names{"id", "name"};

    milvus::EntityRow row;
    auto status = milvus::GetRowFromFieldsData(fields, 1, output_names, row);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(row["id"], 20);
    EXPECT_EQ(row["name"], "b");

    // out of bound
    status = milvus::GetRowFromFieldsData(fields, 10, output_names, row);
    EXPECT_FALSE(status.IsOk());
}

TEST_F(DqlUtilsTest, CopyFieldsDataTest) {
    auto f1 = std::make_shared<milvus::Int64FieldData>("id", std::vector<int64_t>{1, 2, 3, 4, 5});
    auto f2 = std::make_shared<milvus::VarCharFieldData>("name", std::vector<std::string>{"a", "b", "c", "d", "e"});

    std::vector<milvus::FieldDataPtr> src{f1, f2};
    std::vector<milvus::FieldDataPtr> target;
    auto status = milvus::CopyFieldsData(src, 1, 3, target);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(target.size(), 2u);

    auto t1 = std::dynamic_pointer_cast<milvus::Int64FieldData>(target[0]);
    EXPECT_EQ(t1->Count(), 2u);
    EXPECT_EQ(t1->Data()[0], 2);
    EXPECT_EQ(t1->Data()[1], 3);

    auto t2 = std::dynamic_pointer_cast<milvus::VarCharFieldData>(target[1]);
    EXPECT_EQ(t2->Count(), 2u);
    EXPECT_EQ(t2->Data()[0], "b");
    EXPECT_EQ(t2->Data()[1], "c");
}

TEST_F(DqlUtilsTest, AppendFieldDataTest) {
    auto from = std::make_shared<milvus::Int64FieldData>("id", std::vector<int64_t>{4, 5});
    milvus::FieldDataPtr to = std::make_shared<milvus::Int64FieldData>("id", std::vector<int64_t>{1, 2, 3});

    auto status = milvus::AppendFieldData(from, to);
    EXPECT_TRUE(status.IsOk());

    auto result = std::dynamic_pointer_cast<milvus::Int64FieldData>(to);
    EXPECT_EQ(result->Count(), 5u);
    EXPECT_THAT(result->Data(), ElementsAre(1, 2, 3, 4, 5));
}

TEST_F(DqlUtilsTest, AppendFieldDataNullptr) {
    milvus::FieldDataPtr null_ptr;
    milvus::FieldDataPtr valid = std::make_shared<milvus::Int64FieldData>("id", std::vector<int64_t>{1});

    auto status = milvus::AppendFieldData(null_ptr, valid);
    EXPECT_FALSE(status.IsOk());

    status = milvus::AppendFieldData(valid, null_ptr);
    EXPECT_FALSE(status.IsOk());
}

TEST_F(DqlUtilsTest, AppendFieldDataTypeMismatch) {
    auto int_field = std::make_shared<milvus::Int64FieldData>("id", std::vector<int64_t>{1});
    milvus::FieldDataPtr str_field = std::make_shared<milvus::VarCharFieldData>("name", std::vector<std::string>{"a"});

    auto status = milvus::AppendFieldData(int_field, str_field);
    EXPECT_FALSE(status.IsOk());
}

TEST_F(DqlUtilsTest, AppendSearchResultTest) {
    // Build two SingleResult objects and append
    auto ids1 = std::make_shared<milvus::Int64FieldData>("pk", std::vector<int64_t>{1, 2});
    auto scores1 = std::make_shared<milvus::FloatFieldData>("score", std::vector<float>{0.9f, 0.8f});
    std::set<std::string> output_names{"pk", "score"};
    std::vector<milvus::FieldDataPtr> fields1{ids1, scores1};
    milvus::SingleResult sr1("pk", "score", std::move(fields1), output_names);

    auto ids2 = std::make_shared<milvus::Int64FieldData>("pk", std::vector<int64_t>{3});
    auto scores2 = std::make_shared<milvus::FloatFieldData>("score", std::vector<float>{0.7f});
    std::vector<milvus::FieldDataPtr> fields2{ids2, scores2};
    milvus::SingleResult sr2("pk", "score", std::move(fields2), output_names);

    auto status = milvus::AppendSearchResult(sr2, sr1);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(sr1.GetRowCount(), 3u);
}

TEST_F(DqlUtilsTest, AppendSearchResultEmptyTarget) {
    auto ids = std::make_shared<milvus::Int64FieldData>("pk", std::vector<int64_t>{1, 2});
    auto scores = std::make_shared<milvus::FloatFieldData>("score", std::vector<float>{0.9f, 0.8f});
    std::set<std::string> output_names{"pk", "score"};
    std::vector<milvus::FieldDataPtr> fields{ids, scores};
    milvus::SingleResult from("pk", "score", std::move(fields), output_names);

    milvus::SingleResult to;
    auto status = milvus::AppendSearchResult(from, to);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(to.GetRowCount(), 2u);
}

TEST_F(DqlUtilsTest, IsAmbiguousParamTest) {
    // These param names are ambiguous and should return error
    EXPECT_FALSE(milvus::IsAmbiguousParam("params").IsOk());
    EXPECT_FALSE(milvus::IsAmbiguousParam("topk").IsOk());
    EXPECT_FALSE(milvus::IsAmbiguousParam("anns_field").IsOk());
    EXPECT_FALSE(milvus::IsAmbiguousParam("metric_type").IsOk());

    // Non-ambiguous param names should return ok
    EXPECT_TRUE(milvus::IsAmbiguousParam("nprobe").IsOk());
    EXPECT_TRUE(milvus::IsAmbiguousParam("offset").IsOk());
    EXPECT_TRUE(milvus::IsAmbiguousParam("radius").IsOk());
    EXPECT_TRUE(milvus::IsAmbiguousParam("custom_param").IsOk());
}

TEST_F(DqlUtilsTest, ConvertSearchRequestBasic) {
    milvus::SearchArguments args;
    args.SetCollectionName("test_coll");
    args.AddFloatVector({1.0f, 2.0f, 3.0f});
    args.SetLimit(10);
    args.SetFilter("id > 100");

    milvus::proto::milvus::SearchRequest rpc_request;
    auto status = milvus::ConvertSearchRequest(args, "test_db", rpc_request);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(rpc_request.collection_name(), "test_coll");
    EXPECT_EQ(rpc_request.db_name(), "test_db");
    EXPECT_EQ(rpc_request.dsl(), "id > 100");
}

TEST_F(DqlUtilsTest, ConvertQueryRequestBasic) {
    milvus::QueryArguments args;
    args.SetCollectionName("test_coll");
    args.SetFilter("id > 0");
    args.AddOutputField("name");
    args.SetLimit(100);

    milvus::proto::milvus::QueryRequest rpc_request;
    auto status = milvus::ConvertQueryRequest(args, "test_db", rpc_request);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(rpc_request.collection_name(), "test_coll");
    EXPECT_EQ(rpc_request.expr(), "id > 0");
    EXPECT_GE(rpc_request.output_fields_size(), 1);
    EXPECT_EQ(rpc_request.output_fields(0), "name");
}

TEST_F(DqlUtilsTest, ConvertFilterTemplatesTest) {
    std::unordered_map<std::string, nlohmann::json> templates;
    templates["age"] = 25;
    templates["name"] = "alice";
    templates["flag"] = true;
    templates["score"] = 3.14;
    templates["ids"] = nlohmann::json::array({1, 2, 3});

    ::google::protobuf::Map<std::string, milvus::proto::schema::TemplateValue> rpc_templates;
    auto status = milvus::ConvertFilterTemplates(templates, &rpc_templates);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(rpc_templates.size(), 5);

    EXPECT_EQ(rpc_templates["age"].int64_val(), 25);
    EXPECT_EQ(rpc_templates["name"].string_val(), "alice");
    EXPECT_EQ(rpc_templates["flag"].bool_val(), true);
    EXPECT_DOUBLE_EQ(rpc_templates["score"].float_val(), 3.14);
    EXPECT_TRUE(rpc_templates["ids"].has_array_val());
}

TEST_F(DqlUtilsTest, ConvertFilterTemplatesEmpty) {
    std::unordered_map<std::string, nlohmann::json> templates;
    ::google::protobuf::Map<std::string, milvus::proto::schema::TemplateValue> rpc_templates;
    auto status = milvus::ConvertFilterTemplates(templates, &rpc_templates);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(rpc_templates.size(), 0);
}

TEST_F(DqlUtilsTest, ConvertSearchRequestV2) {
    // test ConvertSearchRequest<SearchRequest> template instantiation
    milvus::SearchRequest req;
    req.WithCollectionName("test_coll");
    req.WithAnnsField("vec");
    req.WithLimit(10);
    req.WithMetricType(milvus::MetricType::L2);
    req.AddFloatVector({0.1f, 0.2f, 0.3f});
    req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    req.WithFilter("id > 0");
    req.AddOutputField("name");

    milvus::proto::milvus::SearchRequest rpc_request;
    auto status = milvus::ConvertSearchRequest(req, "default", rpc_request);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(rpc_request.collection_name(), "test_coll");
}

TEST_F(DqlUtilsTest, ConvertSearchRequestWithRange) {
    milvus::SearchRequest req;
    req.WithCollectionName("test_coll");
    req.WithAnnsField("vec");
    req.WithLimit(10);
    req.AddFloatVector({0.1f, 0.2f, 0.3f});
    req.WithRangeFilter(0.3);
    req.WithRadius(1.0);
    req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::proto::milvus::SearchRequest rpc_request;
    auto status = milvus::ConvertSearchRequest(req, "default", rpc_request);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(DqlUtilsTest, ConvertQueryRequestV2) {
    // test ConvertQueryRequest<QueryRequest> template instantiation
    milvus::QueryRequest req;
    req.WithCollectionName("test_coll");
    req.WithFilter("id > 0");
    req.AddOutputField("name");
    req.WithLimit(100);
    req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::proto::milvus::QueryRequest rpc_request;
    auto status = milvus::ConvertQueryRequest(req, "default", rpc_request);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(rpc_request.collection_name(), "test_coll");
    EXPECT_EQ(rpc_request.expr(), "id > 0");
}

TEST_F(DqlUtilsTest, ConvertQueryIteratorRequest) {
    // test ConvertQueryRequest<QueryIteratorRequest> template instantiation
    milvus::QueryIteratorRequest req;
    req.WithCollectionName("test_coll");
    req.WithFilter("id >= 0");
    req.AddOutputField("name");
    req.SetBatchSize(100);
    req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::proto::milvus::QueryRequest rpc_request;
    auto status = milvus::ConvertQueryRequest(req, "default", rpc_request);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(DqlUtilsTest, ConvertSearchIteratorRequest) {
    // test ConvertSearchRequest<SearchIteratorRequest> template instantiation
    milvus::SearchIteratorRequest req;
    req.WithCollectionName("test_coll");
    req.WithAnnsField("vec");
    req.WithLimit(100);
    req.SetBatchSize(50);
    req.AddFloatVector({0.1f, 0.2f, 0.3f});
    req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::proto::milvus::SearchRequest rpc_request;
    auto status = milvus::ConvertSearchRequest(req, "default", rpc_request);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(DqlUtilsTest, ConvertHybridSearchRequestV2) {
    // test ConvertHybridSearchRequest<HybridSearchRequest> template instantiation
    auto sub1 = std::make_shared<milvus::SubSearchRequest>();
    sub1->WithAnnsField("vec1").WithLimit(5).AddFloatVector({0.1f, 0.2f, 0.3f});

    auto sub2 = std::make_shared<milvus::SubSearchRequest>();
    sub2->WithAnnsField("vec2").WithLimit(5).AddFloatVector({0.4f, 0.5f, 0.6f});

    milvus::HybridSearchRequest req;
    req.WithCollectionName("test_coll");
    req.AddSubRequest(sub1);
    req.AddSubRequest(sub2);
    req.WithRerank(std::make_shared<milvus::RRFRerank>(60));
    req.WithLimit(10);
    req.WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::proto::milvus::HybridSearchRequest rpc_request;
    auto status = milvus::ConvertHybridSearchRequest(req, "default", rpc_request);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(rpc_request.collection_name(), "test_coll");
}

TEST_F(DqlUtilsTest, AppendFieldDataAllTypes) {
    // test Append<> template for various field types not covered by the basic test

    // VarChar
    {
        auto src = std::make_shared<milvus::VarCharFieldData>("name", std::vector<std::string>{"a", "b"});
        milvus::FieldDataPtr target = std::make_shared<milvus::VarCharFieldData>("name");
        auto status = milvus::AppendFieldData(src, target);
        EXPECT_TRUE(status.IsOk());
        EXPECT_EQ(target->Count(), 2);
    }

    // JSON
    {
        auto src =
            std::make_shared<milvus::JSONFieldData>("json", std::vector<nlohmann::json>{nlohmann::json{{"k", "v"}}});
        milvus::FieldDataPtr target = std::make_shared<milvus::JSONFieldData>("json");
        auto status = milvus::AppendFieldData(src, target);
        EXPECT_TRUE(status.IsOk());
        EXPECT_EQ(target->Count(), 1);
    }

    // BinaryVec
    {
        auto src = std::make_shared<milvus::BinaryVecFieldData>("bin", std::vector<std::vector<uint8_t>>{{0xFF, 0x00}});
        milvus::FieldDataPtr target = std::make_shared<milvus::BinaryVecFieldData>("bin");
        auto status = milvus::AppendFieldData(src, target);
        EXPECT_TRUE(status.IsOk());
        EXPECT_EQ(target->Count(), 1);
    }

    // Array (Int32)
    {
        auto src = std::make_shared<milvus::ArrayInt32FieldData>("arr", std::vector<std::vector<int32_t>>{{1, 2}, {3}});
        milvus::FieldDataPtr target = std::make_shared<milvus::ArrayInt32FieldData>("arr");
        auto status = milvus::AppendFieldData(src, target);
        EXPECT_TRUE(status.IsOk());
        EXPECT_EQ(target->Count(), 2);
    }
}

TEST_F(DqlUtilsTest, CopyFieldDataRangeAllTypes) {
    // test CopyFieldData for types not covered by the basic test

    // VarChar
    {
        auto src = std::make_shared<milvus::VarCharFieldData>("name", std::vector<std::string>{"a", "b", "c"});
        milvus::FieldDataPtr target;
        auto status = milvus::CopyFieldData(src, 1, 3, target);
        EXPECT_TRUE(status.IsOk());
        EXPECT_EQ(target->Count(), 2);
    }

    // Bool
    {
        auto src = std::make_shared<milvus::BoolFieldData>("flag", std::vector<bool>{true, false, true});
        milvus::FieldDataPtr target;
        auto status = milvus::CopyFieldData(src, 0, 2, target);
        EXPECT_TRUE(status.IsOk());
        EXPECT_EQ(target->Count(), 2);
    }

    // FloatVec
    {
        auto src = std::make_shared<milvus::FloatVecFieldData>(
            "vec", std::vector<std::vector<float>>{{0.1f, 0.2f}, {0.3f, 0.4f}, {0.5f, 0.6f}});
        milvus::FieldDataPtr target;
        auto status = milvus::CopyFieldData(src, 0, 2, target);
        EXPECT_TRUE(status.IsOk());
        EXPECT_EQ(target->Count(), 2);
    }

    // BinaryVec
    {
        auto src = std::make_shared<milvus::BinaryVecFieldData>(
            "bin", std::vector<std::vector<uint8_t>>{{0xFF}, {0x00}, {0xAB}});
        milvus::FieldDataPtr target;
        auto status = milvus::CopyFieldData(src, 1, 3, target);
        EXPECT_TRUE(status.IsOk());
        EXPECT_EQ(target->Count(), 2);
    }
}

TEST_F(DqlUtilsTest, SetTargetVectorsFloat) {
    auto vectors = std::make_shared<milvus::FloatVecFieldData>(
        "vec", std::vector<std::vector<float>>{{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}});

    milvus::proto::milvus::SearchRequest rpc_request;
    auto status = milvus::SetTargetVectors(vectors, &rpc_request);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(rpc_request.nq(), 2);
}

TEST_F(DqlUtilsTest, SetTargetVectorsBinary) {
    auto vectors = std::make_shared<milvus::BinaryVecFieldData>(
        "bin", std::vector<std::vector<uint8_t>>{{0xFF, 0x00}, {0xAB, 0xCD}});

    milvus::proto::milvus::SearchRequest rpc_request;
    auto status = milvus::SetTargetVectors(vectors, &rpc_request);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(rpc_request.nq(), 2);
}

TEST_F(DqlUtilsTest, SetTargetVectorsSparse) {
    auto vectors = std::make_shared<milvus::SparseFloatVecFieldData>(
        "sp", std::vector<std::map<uint32_t, float>>{{{1, 0.5f}}, {{2, 0.3f}}});

    milvus::proto::milvus::SearchRequest rpc_request;
    auto status = milvus::SetTargetVectors(vectors, &rpc_request);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(rpc_request.nq(), 2);
}

TEST_F(DqlUtilsTest, SetExtraParamsTest) {
    std::unordered_map<std::string, std::string> params = {{"nprobe", "10"}, {"ef", "200"}};
    ::google::protobuf::RepeatedPtrField<milvus::proto::common::KeyValuePair> kv_pairs;
    milvus::SetExtraParams(params, &kv_pairs);
    EXPECT_GE(kv_pairs.size(), 2);
}
