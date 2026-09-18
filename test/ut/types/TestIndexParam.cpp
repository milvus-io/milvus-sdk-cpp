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

#include "milvus/types/IndexParam.h"

class IndexParamTest : public ::testing::Test {};

TEST_F(IndexParamTest, GeneralTesting) {
    milvus::IndexParam index_param("field_name", "index_name", milvus::IndexType::HNSW, milvus::MetricType::L2);

    EXPECT_EQ(index_param.FieldName(), "field_name");
    EXPECT_EQ(index_param.IndexName(), "index_name");
    EXPECT_EQ(index_param.IndexType(), milvus::IndexType::HNSW);
    EXPECT_EQ(index_param.MetricType(), milvus::MetricType::L2);
}

TEST_F(IndexParamTest, SettersAndGetters) {
    milvus::IndexParam index_param;

    index_param.SetFieldName("field_name");
    EXPECT_EQ(index_param.FieldName(), "field_name");

    index_param.SetIndexName("index_name");
    EXPECT_EQ(index_param.IndexName(), "index_name");

    index_param.SetIndexType(milvus::IndexType::IVF_FLAT);
    EXPECT_EQ(index_param.IndexType(), milvus::IndexType::IVF_FLAT);

    index_param.SetMetricType(milvus::MetricType::COSINE);
    EXPECT_EQ(index_param.MetricType(), milvus::MetricType::COSINE);
}

TEST_F(IndexParamTest, AddExtraParam) {
    milvus::IndexParam index_param;
    auto status = index_param.AddExtraParam("nlist", "1024");
    EXPECT_TRUE(status.IsOk());

    status = index_param.AddExtraParam("nprobe", "16");
    EXPECT_TRUE(status.IsOk());

    auto& params = index_param.ExtraParams();
    EXPECT_EQ(params.size(), 2);
    EXPECT_EQ(params.at("nlist"), "1024");
    EXPECT_EQ(params.at("nprobe"), "16");
}

TEST_F(IndexParamTest, ExtraParamsFromJson) {
    milvus::IndexParam index_param;
    auto status = index_param.ExtraParamsFromJson(R"({"nlist":"1024","nprobe":"16"})");
    EXPECT_TRUE(status.IsOk());

    auto& params = index_param.ExtraParams();
    EXPECT_EQ(params.at("nlist"), "1024");
    EXPECT_EQ(params.at("nprobe"), "16");

    // invalid json
    status = index_param.ExtraParamsFromJson("not valid json");
    EXPECT_FALSE(status.IsOk());
}
