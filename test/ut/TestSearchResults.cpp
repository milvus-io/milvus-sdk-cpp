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

#include "types/SearchResults.h"

class SearchResultsTest : public ::testing::Test {};

TEST_F(SearchResultsTest, GeneralTesting) {
    milvus::IDArray ids{std::vector<int64_t>{}};
    std::vector<float> scores{};
    std::vector<milvus::FieldDataPtr> fields{};

    std::vector<milvus::SingleResult> result_array = {
        milvus::SingleResult(std::move(ids), std::move(scores), std::move(fields))};

    milvus::SearchResults results(std::move(result_array));
    EXPECT_EQ(1, results.Results().size());
}
