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
