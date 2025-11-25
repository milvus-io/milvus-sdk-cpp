
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

#include "milvus/types/DmlResults.h"

class DmlResultsTest : public ::testing::Test {};

TEST_F(DmlResultsTest, GeneralTesting) {
    milvus::DmlResults dml_results;
    milvus::IDArray id_array{std::vector<int64_t>{10000, 20000}};
    dml_results.SetIdArray(id_array);
    dml_results.SetTimestamp(10000);
    EXPECT_EQ(dml_results.IdArray().IntIDArray(), id_array.IntIDArray());
    EXPECT_EQ(dml_results.Timestamp(), 10000);
}
