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

#include "milvus/MilvusClientV2.h"

class DataTypeEnumTest : public ::testing::Test {};

TEST_F(DataTypeEnumTest, KnownValues) {
    EXPECT_EQ(static_cast<int>(milvus::DataType::UNKNOWN), 0);
    EXPECT_EQ(static_cast<int>(milvus::DataType::BOOL), 1);
    EXPECT_EQ(static_cast<int>(milvus::DataType::INT8), 2);
    EXPECT_EQ(static_cast<int>(milvus::DataType::INT16), 3);
    EXPECT_EQ(static_cast<int>(milvus::DataType::INT32), 4);
    EXPECT_EQ(static_cast<int>(milvus::DataType::INT64), 5);
    EXPECT_EQ(static_cast<int>(milvus::DataType::FLOAT), 10);
    EXPECT_EQ(static_cast<int>(milvus::DataType::DOUBLE), 11);
    EXPECT_EQ(static_cast<int>(milvus::DataType::VARCHAR), 21);
    EXPECT_EQ(static_cast<int>(milvus::DataType::ARRAY), 22);
    EXPECT_EQ(static_cast<int>(milvus::DataType::JSON), 23);
    EXPECT_EQ(static_cast<int>(milvus::DataType::BINARY_VECTOR), 100);
    EXPECT_EQ(static_cast<int>(milvus::DataType::FLOAT_VECTOR), 101);
    EXPECT_EQ(static_cast<int>(milvus::DataType::FLOAT16_VECTOR), 102);
    EXPECT_EQ(static_cast<int>(milvus::DataType::BFLOAT16_VECTOR), 103);
    EXPECT_EQ(static_cast<int>(milvus::DataType::SPARSE_FLOAT_VECTOR), 104);
    EXPECT_EQ(static_cast<int>(milvus::DataType::INT8_VECTOR), 105);
    EXPECT_EQ(static_cast<int>(milvus::DataType::STRUCT), 201);
}

TEST_F(DataTypeEnumTest, Comparison) {
    EXPECT_NE(milvus::DataType::INT32, milvus::DataType::INT64);
    EXPECT_EQ(milvus::DataType::FLOAT, milvus::DataType::FLOAT);
}

class ConsistencyLevelEnumTest : public ::testing::Test {};

TEST_F(ConsistencyLevelEnumTest, KnownValues) {
    EXPECT_EQ(static_cast<int>(milvus::ConsistencyLevel::NONE), -1);
    EXPECT_EQ(static_cast<int>(milvus::ConsistencyLevel::STRONG), 0);
    EXPECT_EQ(static_cast<int>(milvus::ConsistencyLevel::SESSION), 1);
    EXPECT_EQ(static_cast<int>(milvus::ConsistencyLevel::BOUNDED), 2);
    EXPECT_EQ(static_cast<int>(milvus::ConsistencyLevel::EVENTUALLY), 3);
}

class IndexTypeEnumTest : public ::testing::Test {};

TEST_F(IndexTypeEnumTest, KnownValues) {
    EXPECT_EQ(static_cast<int>(milvus::IndexType::INVALID), 0);
    EXPECT_EQ(static_cast<int>(milvus::IndexType::FLAT), 1);
    EXPECT_EQ(static_cast<int>(milvus::IndexType::IVF_FLAT), 2);
    EXPECT_EQ(static_cast<int>(milvus::IndexType::HNSW), 5);
    EXPECT_EQ(static_cast<int>(milvus::IndexType::DISKANN), 6);
    EXPECT_EQ(static_cast<int>(milvus::IndexType::AUTOINDEX), 7);
    EXPECT_EQ(static_cast<int>(milvus::IndexType::AISAQ), 13);
    EXPECT_EQ(static_cast<int>(milvus::IndexType::BIN_FLAT), 1001);
    EXPECT_EQ(static_cast<int>(milvus::IndexType::RTREE), 1301);
    EXPECT_EQ(static_cast<int>(milvus::IndexType::SPARSE_INVERTED_INDEX), 1201);
}

class MetricTypeEnumTest : public ::testing::Test {};

TEST_F(MetricTypeEnumTest, KnownValues) {
    EXPECT_EQ(static_cast<int>(milvus::MetricType::DEFAULT), 0);
    EXPECT_EQ(static_cast<int>(milvus::MetricType::L2), 1);
    EXPECT_EQ(static_cast<int>(milvus::MetricType::IP), 2);
    EXPECT_EQ(static_cast<int>(milvus::MetricType::COSINE), 3);
    EXPECT_EQ(static_cast<int>(milvus::MetricType::HAMMING), 101);
    EXPECT_EQ(static_cast<int>(milvus::MetricType::JACCARD), 102);
    EXPECT_EQ(static_cast<int>(milvus::MetricType::BM25), 201);
}

TEST_F(MetricTypeEnumTest, InvalidIsDefault) {
    EXPECT_EQ(milvus::MetricType::INVALID, milvus::MetricType::DEFAULT);
}

class IndexStateEnumTest : public ::testing::Test {};

TEST_F(IndexStateEnumTest, IndexStateCodeValues) {
    EXPECT_EQ(static_cast<int>(milvus::IndexStateCode::NONE), 0);
    EXPECT_EQ(static_cast<int>(milvus::IndexStateCode::UNISSUED), 1);
    EXPECT_EQ(static_cast<int>(milvus::IndexStateCode::IN_PROGRESS), 2);
    EXPECT_EQ(static_cast<int>(milvus::IndexStateCode::FINISHED), 3);
    EXPECT_EQ(static_cast<int>(milvus::IndexStateCode::FAILED), 4);
    EXPECT_EQ(static_cast<int>(milvus::IndexStateCode::RETRY), 5);
}

class LoadStateEnumTest : public ::testing::Test {};

TEST_F(LoadStateEnumTest, KnownValues) {
    EXPECT_EQ(static_cast<int>(milvus::LoadState::LOAD_STATE_NOT_EXIST), 0);
    EXPECT_EQ(static_cast<int>(milvus::LoadState::LOAD_STATE_NOT_LOAD), 1);
    EXPECT_EQ(static_cast<int>(milvus::LoadState::LOAD_STATE_LOADING), 2);
    EXPECT_EQ(static_cast<int>(milvus::LoadState::LOAD_STATE_LOADED), 3);
}

class FunctionTypeEnumTest : public ::testing::Test {};

TEST_F(FunctionTypeEnumTest, KnownValues) {
    EXPECT_EQ(static_cast<int>(milvus::FunctionType::UNKNOWN), 0);
    EXPECT_EQ(static_cast<int>(milvus::FunctionType::BM25), 1);
    EXPECT_EQ(static_cast<int>(milvus::FunctionType::TEXTEMBEDDING), 2);
    EXPECT_EQ(static_cast<int>(milvus::FunctionType::RERANK), 3);
}
