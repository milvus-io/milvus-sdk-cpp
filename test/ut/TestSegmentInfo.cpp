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

#include "milvus/types/SegmentInfo.h"

class SegmentInfoTest : public ::testing::Test {};

TEST_F(SegmentInfoTest, GeneralTesting) {
    int64_t collection_id = 1;
    int64_t partition_id = 2;
    int64_t segment_id = 3;
    int64_t row_count = 4;

    milvus::SegmentState state = milvus::SegmentState::GROWING;

    milvus::SegmentInfo info(collection_id, partition_id, segment_id, row_count, state);
    EXPECT_EQ(collection_id, info.CollectionID());
    EXPECT_EQ(partition_id, info.PartitionID());
    EXPECT_EQ(segment_id, info.SegmentID());
    EXPECT_EQ(row_count, info.RowCount());
    EXPECT_EQ(state, info.State());

    std::string index_name = "IVF_FLAT";
    int64_t index_id = 5;
    int64_t node_id = 6;
    milvus::QuerySegmentInfo query_info(collection_id, partition_id, segment_id, row_count, state, index_name, index_id,
                                        node_id);
    EXPECT_EQ(index_name, query_info.IndexName());
    EXPECT_EQ(index_id, query_info.IndexID());
    EXPECT_EQ(node_id, query_info.NodeID());
}
