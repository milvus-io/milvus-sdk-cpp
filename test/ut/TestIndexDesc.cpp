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

#include "milvus/types/IndexDesc.h"

class IndexDescTest : public ::testing::Test {};

TEST_F(IndexDescTest, GeneralTesting) {
    milvus::IndexDesc index_desc;

    index_desc.SetFieldName("field_name");
    EXPECT_EQ(index_desc.FieldName(), "field_name");

    index_desc.SetIndexName("index_name");
    EXPECT_EQ(index_desc.IndexName(), "index_name");

    index_desc.SetIndexId(1);
    EXPECT_EQ(index_desc.IndexId(), 1);

    index_desc.SetIndexType(milvus::IndexType::IVF_FLAT);
    EXPECT_EQ(index_desc.IndexType(), milvus::IndexType::IVF_FLAT);

    index_desc.SetMetricType(milvus::MetricType::L2);
    EXPECT_EQ(index_desc.MetricType(), milvus::MetricType::L2);
}

TEST_F(IndexDescTest, ValidationTestingPossitive) {
    for (const auto& test_data : {
             std::make_tuple(milvus::IndexType::IVF_FLAT, milvus::MetricType::L2, "{\"nlist\": 1024}"),
             std::make_tuple(milvus::IndexType::IVF_SQ8, milvus::MetricType::L2, "{\"nlist\": 1024}"),
             std::make_tuple(milvus::IndexType::IVF_PQ, milvus::MetricType::L2, "{\"nlist\": 1024, \"m\": 8}"),
             std::make_tuple(milvus::IndexType::IVF_PQ, milvus::MetricType::L2,
                             "{\"nlist\": 1024, \"m\": 8, \"nbits\": 12}"),
             std::make_tuple(milvus::IndexType::HNSW, milvus::MetricType::L2, "{\"M\": 4, \"efConstruction\": 12}"),
             std::make_tuple(milvus::IndexType::IVF_HNSW, milvus::MetricType::L2,
                             "{\"nlist\": 1024, \"M\": 4, \"efConstruction\": 12}"),
             std::make_tuple(milvus::IndexType::RHNSW_PQ, milvus::MetricType::L2,
                             "{\"PQM\": 32, \"M\": 4, \"efConstruction\": 12}"),
             std::make_tuple(milvus::IndexType::ANNOY, milvus::MetricType::L2, "{\"n_trees\": 12}"),
             std::make_tuple(milvus::IndexType::BIN_IVF_FLAT, milvus::MetricType::JACCARD, ""),
             std::make_tuple(milvus::IndexType::BIN_IVF_FLAT, milvus::MetricType::TANIMOTO, ""),
             std::make_tuple(milvus::IndexType::BIN_IVF_FLAT, milvus::MetricType::HAMMING, ""),
             std::make_tuple(milvus::IndexType::BIN_FLAT, milvus::MetricType::SUBSTRUCTURE, ""),
             std::make_tuple(milvus::IndexType::BIN_FLAT, milvus::MetricType::SUPERSTRUCTURE, ""),
         }) {
        milvus::IndexDesc index_desc{"foo", "index", std::get<0>(test_data), std::get<1>(test_data), 0};
        index_desc.ExtraParamsFromJson(std::get<2>(test_data));
        auto status = index_desc.Validate();
        EXPECT_TRUE(status.IsOk());
        EXPECT_EQ(status.Message(), "OK");
    }
}

TEST_F(IndexDescTest, ValidationTestingNegative) {
    for (const auto& test_data : {
             std::make_tuple(milvus::IndexType::IVF_FLAT, milvus::MetricType::JACCARD, "{\"nlist\": 1024}"),
             std::make_tuple(milvus::IndexType::BIN_IVF_FLAT, milvus::MetricType::SUBSTRUCTURE, "{\"nlist\": 1024}"),
             std::make_tuple(milvus::IndexType::IVF_FLAT, milvus::MetricType::IP, "{\"nlist\": 0}"),
             std::make_tuple(milvus::IndexType::IVF_FLAT, milvus::MetricType::IP, "{\"nlist\": \"1024\"}"),
             std::make_tuple(milvus::IndexType::IVF_FLAT, milvus::MetricType::IP, "{\"nlist\": 65537}"),
             std::make_tuple(milvus::IndexType::ANNOY, milvus::MetricType::IP, "{\"nlist\": 65537}"),
             std::make_tuple(milvus::IndexType::BIN_IVF_FLAT, milvus::MetricType::IP, "{\"nlist\": 1024}"),
         }) {
        milvus::IndexDesc index_desc{"foo", "index", std::get<0>(test_data), std::get<1>(test_data), 0};
        index_desc.ExtraParamsFromJson(std::get<2>(test_data));
        auto status = index_desc.Validate();
        EXPECT_FALSE(status.IsOk());
    }
}
