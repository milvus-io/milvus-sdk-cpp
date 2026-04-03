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

class EmbeddingListTest : public ::testing::Test {};

TEST_F(EmbeddingListTest, DefaultConstructor) {
    milvus::EmbeddingList emb;
    EXPECT_EQ(emb.Count(), 0);
    EXPECT_EQ(emb.Dim(), 0);
    EXPECT_EQ(emb.TargetVectors(), nullptr);
}

TEST_F(EmbeddingListTest, AddFloatVector) {
    milvus::EmbeddingList emb;
    std::vector<float> vec = {1.0f, 2.0f, 3.0f};
    auto status = emb.AddFloatVector(vec);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 1);
    EXPECT_EQ(emb.Dim(), 3);
}

TEST_F(EmbeddingListTest, AddMultipleFloatVectors) {
    milvus::EmbeddingList emb;
    std::vector<float> vec1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> vec2 = {4.0f, 5.0f, 6.0f};
    EXPECT_TRUE(emb.AddFloatVector(vec1).IsOk());
    EXPECT_TRUE(emb.AddFloatVector(vec2).IsOk());
    EXPECT_EQ(emb.Count(), 2);
    EXPECT_EQ(emb.Dim(), 3);
}

TEST_F(EmbeddingListTest, AddBinaryVector) {
    milvus::EmbeddingList emb;
    std::vector<uint8_t> bin_vec = {0xFF, 0x00, 0xAB};
    auto status = emb.AddBinaryVector(bin_vec);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 1);
}

TEST_F(EmbeddingListTest, AddSparseVectorJson) {
    milvus::EmbeddingList emb;
    nlohmann::json sparse = {{"1", 0.1}, {"5", 0.2}, {"8", 0.15}};
    auto status = emb.AddSparseVector(sparse);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 1);
}

TEST_F(EmbeddingListTest, SetFloatVectors) {
    milvus::EmbeddingList emb;
    std::vector<std::vector<float>> vecs = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto status = emb.SetFloatVectors(std::move(vecs));
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 2);
    EXPECT_EQ(emb.Dim(), 2);
}

TEST_F(EmbeddingListTest, AddEmbeddedText) {
    milvus::EmbeddingList emb;
    auto status = emb.AddEmbeddedText("hello world");
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 1);
}

TEST_F(EmbeddingListTest, AddFloat16VectorFromFloats) {
    milvus::EmbeddingList emb;
    std::vector<float> vec = {1.0f, 2.0f, 3.0f};
    auto status = emb.AddFloat16Vector(vec);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 1);
    EXPECT_EQ(emb.Dim(), 3);
}

TEST_F(EmbeddingListTest, AddFloat16VectorFromBinary) {
    milvus::EmbeddingList emb;
    // 3 dimensions = 3 uint16_t elements
    std::vector<uint16_t> vec = {0x3C00, 0x4000, 0x4200};
    auto status = emb.AddFloat16Vector(vec);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 1);
    EXPECT_EQ(emb.Dim(), 3);
}

TEST_F(EmbeddingListTest, AddBFloat16VectorFromFloats) {
    milvus::EmbeddingList emb;
    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f};
    auto status = emb.AddBFloat16Vector(vec);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 1);
    EXPECT_EQ(emb.Dim(), 4);
}

TEST_F(EmbeddingListTest, AddBFloat16VectorFromBinary) {
    milvus::EmbeddingList emb;
    std::vector<uint16_t> vec = {0x3F80, 0x4000};
    auto status = emb.AddBFloat16Vector(vec);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 1);
    EXPECT_EQ(emb.Dim(), 2);
}

TEST_F(EmbeddingListTest, AddInt8Vector) {
    milvus::EmbeddingList emb;
    std::vector<int8_t> vec = {1, -2, 3, -4};
    auto status = emb.AddInt8Vector(vec);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 1);
    EXPECT_EQ(emb.Dim(), 4);
}

TEST_F(EmbeddingListTest, AddBinaryVectorFromString) {
    milvus::EmbeddingList emb;
    std::string bin_str = "\xFF\x00\xAB\xCD";
    auto status = emb.AddBinaryVector(bin_str);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 1);
}

TEST_F(EmbeddingListTest, AddSparseVectorFromMap) {
    milvus::EmbeddingList emb;
    std::map<uint32_t, float> sparse = {{1, 0.1f}, {5, 0.2f}, {8, 0.15f}};
    auto status = emb.AddSparseVector(sparse);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 1);
}

TEST_F(EmbeddingListTest, SetBinaryVectorsFromStrings) {
    milvus::EmbeddingList emb;
    // binary vectors as strings: each string represents raw bytes
    // use strings with consistent length (dim/8 bytes each)
    std::string v1(4, '\xFF');  // 4 bytes = 32 dim
    std::string v2(4, '\x00');
    std::vector<std::string> vecs = {v1, v2};
    auto status = emb.SetBinaryVectors(vecs);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 2);
}

TEST_F(EmbeddingListTest, SetBinaryVectorsFromUint8) {
    milvus::EmbeddingList emb;
    std::vector<std::vector<uint8_t>> vecs = {{0xFF, 0x00}, {0xAB, 0xCD}};
    auto status = emb.SetBinaryVectors(std::move(vecs));
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 2);
}

TEST_F(EmbeddingListTest, SetSparseVectorsFromMap) {
    milvus::EmbeddingList emb;
    std::vector<std::map<uint32_t, float>> vecs = {{{1, 0.1f}}, {{2, 0.2f}, {3, 0.3f}}};
    auto status = emb.SetSparseVectors(std::move(vecs));
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 2);
}

TEST_F(EmbeddingListTest, SetSparseVectorsFromJson) {
    milvus::EmbeddingList emb;
    std::vector<nlohmann::json> vecs = {
        nlohmann::json{{"1", 0.1}, {"5", 0.2}},
        nlohmann::json{{"2", 0.3}},
    };
    auto status = emb.SetSparseVectors(vecs);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 2);
}

TEST_F(EmbeddingListTest, SetFloat16VectorsFromBinary) {
    milvus::EmbeddingList emb;
    std::vector<std::vector<uint16_t>> vecs = {{0x3C00, 0x4000}, {0x4200, 0x4400}};
    auto status = emb.SetFloat16Vectors(std::move(vecs));
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 2);
    EXPECT_EQ(emb.Dim(), 2);
}

TEST_F(EmbeddingListTest, SetFloat16VectorsFromFloats) {
    milvus::EmbeddingList emb;
    std::vector<std::vector<float>> vecs = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto status = emb.SetFloat16Vectors(vecs);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 2);
    EXPECT_EQ(emb.Dim(), 2);
}

TEST_F(EmbeddingListTest, SetBFloat16VectorsFromBinary) {
    milvus::EmbeddingList emb;
    std::vector<std::vector<uint16_t>> vecs = {{0x3F80, 0x4000}, {0x4040, 0x4080}};
    auto status = emb.SetBFloat16Vectors(std::move(vecs));
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 2);
    EXPECT_EQ(emb.Dim(), 2);
}

TEST_F(EmbeddingListTest, SetBFloat16VectorsFromFloats) {
    milvus::EmbeddingList emb;
    std::vector<std::vector<float>> vecs = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    auto status = emb.SetBFloat16Vectors(vecs);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 2);
    EXPECT_EQ(emb.Dim(), 3);
}

TEST_F(EmbeddingListTest, SetEmbeddedTexts) {
    milvus::EmbeddingList emb;
    std::vector<std::string> texts = {"hello", "world", "milvus"};
    auto status = emb.SetEmbeddedTexts(std::move(texts));
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 3);
}

TEST_F(EmbeddingListTest, SetInt8Vectors) {
    milvus::EmbeddingList emb;
    std::vector<std::vector<int8_t>> vecs = {{1, -2, 3}, {4, -5, 6}};
    auto status = emb.SetInt8Vectors(std::move(vecs));
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(emb.Count(), 2);
    EXPECT_EQ(emb.Dim(), 3);
}

TEST_F(EmbeddingListTest, TypeMismatchError) {
    milvus::EmbeddingList emb;
    // add a float vector first
    EXPECT_TRUE(emb.AddFloatVector({1.0f, 2.0f}).IsOk());
    // then try to add a binary vector — should fail due to type mismatch
    auto status = emb.AddBinaryVector(std::vector<uint8_t>{0xFF, 0x00});
    EXPECT_FALSE(status.IsOk());
}

TEST_F(EmbeddingListTest, SetResetsExisting) {
    milvus::EmbeddingList emb;
    // add some float vectors
    EXPECT_TRUE(emb.AddFloatVector({1.0f, 2.0f}).IsOk());
    EXPECT_TRUE(emb.AddFloatVector({3.0f, 4.0f}).IsOk());
    EXPECT_EQ(emb.Count(), 2);

    // SetFloatVectors should reset
    std::vector<std::vector<float>> new_vecs = {{5.0f, 6.0f}};
    EXPECT_TRUE(emb.SetFloatVectors(std::move(new_vecs)).IsOk());
    EXPECT_EQ(emb.Count(), 1);
}

TEST_F(EmbeddingListTest, AddMultipleSparseVectors) {
    milvus::EmbeddingList emb;
    std::map<uint32_t, float> sparse1 = {{1, 0.1f}, {5, 0.2f}};
    std::map<uint32_t, float> sparse2 = {{2, 0.3f}};
    EXPECT_TRUE(emb.AddSparseVector(sparse1).IsOk());
    EXPECT_TRUE(emb.AddSparseVector(sparse2).IsOk());
    EXPECT_EQ(emb.Count(), 2);
}
