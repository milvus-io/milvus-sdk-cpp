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

#include "milvus/types/SearchResults.h"

class SearchResultsTest : public ::testing::Test {};

TEST_F(SearchResultsTest, TestSingleResult) {
    std::vector<milvus::FieldDataPtr> fields{
        std::make_shared<milvus::Int64FieldData>("pk", std::vector<int64_t>{10000}),
        std::make_shared<milvus::FloatFieldData>("score", std::vector<float>{0.1}),
        std::make_shared<milvus::BoolFieldData>("bool", std::vector<bool>{true}),
        std::make_shared<milvus::Int16FieldData>("int16", std::vector<int16_t>{1})};
    std::set<std::string> output_names;
    output_names.insert("int16");
    milvus::SingleResult result{"pk", "score", std::move(fields), output_names};
    EXPECT_EQ(result.PrimaryKeyName(), "pk");
    EXPECT_EQ(result.Ids().IntIDArray(), std::vector<int64_t>{10000});
    EXPECT_EQ(result.Scores(), std::vector<float>{0.1f});
    EXPECT_EQ(result.OutputField("bool")->Name(), "bool");
    EXPECT_EQ(result.OutputField("int16")->Name(), "int16");
    EXPECT_EQ(result.OutputField("invalid"), nullptr);
    EXPECT_EQ(result.OutputFields().size(), 4);
    EXPECT_EQ(result.OutputFieldNames().size(), output_names.size());
}

TEST_F(SearchResultsTest, GeneralTesting) {
    std::vector<milvus::FieldDataPtr> fields{};
    std::set<std::string> output_names;
    output_names.insert("int16");
    milvus::SingleResult single("id", "distance", std::move(fields), output_names);
    std::vector<milvus::SingleResult> result_array;
    result_array.emplace_back(std::move(single));

    milvus::SearchResults results(std::move(result_array));
    EXPECT_EQ(1, results.Results().size());
}
