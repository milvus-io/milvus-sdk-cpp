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

#include "utils/MiscUtils.h"

class MiscUtilsTest : public ::testing::Test {};

TEST_F(MiscUtilsTest, Trim) {
    EXPECT_EQ(milvus::Trim("value"), "value");
    EXPECT_EQ(milvus::Trim("  value  "), "value");
    EXPECT_EQ(milvus::Trim("\t\nvalue with spaces\r\n"), "value with spaces");
    EXPECT_TRUE(milvus::Trim(" \t\n").empty());
}

TEST_F(MiscUtilsTest, UpperWithoutSpaces) {
    EXPECT_EQ(milvus::UpperWithoutSpaces("mb"), "MB");
    EXPECT_EQ(milvus::UpperWithoutSpaces(" g b "), "GB");
    EXPECT_EQ(milvus::UpperWithoutSpaces("t\tb\n"), "TB");
    EXPECT_EQ(milvus::UpperWithoutSpaces("MiB"), "MIB");
}

TEST_F(MiscUtilsTest, ParseTargetSizeMB) {
    int64_t target_size_mb = -1;
    std::string normalized = "unchanged";

    auto status = milvus::ParseTargetSizeMB("", target_size_mb, normalized);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(target_size_mb, 0);
    EXPECT_TRUE(normalized.empty());

    status = milvus::ParseTargetSizeMB("1048576 B", target_size_mb, normalized);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(target_size_mb, 1);
    EXPECT_EQ(normalized, "1MB");

    status = milvus::ParseTargetSizeMB("1536 KB", target_size_mb, normalized);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(target_size_mb, 1);
    EXPECT_EQ(normalized, "1MB");

    status = milvus::ParseTargetSizeMB(" 1 MB ", target_size_mb, normalized);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(target_size_mb, 1);
    EXPECT_EQ(normalized, "1MB");

    status = milvus::ParseTargetSizeMB("2gb", target_size_mb, normalized);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(target_size_mb, 2048);
    EXPECT_EQ(normalized, "2048MB");

    status = milvus::ParseTargetSizeMB("3TB", target_size_mb, normalized);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(target_size_mb, 3145728);
    EXPECT_EQ(normalized, "3145728MB");

    status = milvus::ParseTargetSizeMB("4PB", target_size_mb, normalized);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(target_size_mb, 4294967296);
    EXPECT_EQ(normalized, "4294967296MB");

    status = milvus::ParseTargetSizeMB("1.5GB", target_size_mb, normalized);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(target_size_mb, 1536);
    EXPECT_EQ(normalized, "1536MB");

    status = milvus::ParseTargetSizeMB("1048576", target_size_mb, normalized);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(target_size_mb, 1);
    EXPECT_EQ(normalized, "1MB");
}

TEST_F(MiscUtilsTest, ParseTargetSizeMBInvalid) {
    int64_t target_size_mb = 0;
    std::string normalized;

    auto status = milvus::ParseTargetSizeMB("1023KB", target_size_mb, normalized);
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);

    status = milvus::ParseTargetSizeMB("0", target_size_mb, normalized);
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);

    status = milvus::ParseTargetSizeMB("-1MB", target_size_mb, normalized);
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);

    status = milvus::ParseTargetSizeMB("1XB", target_size_mb, normalized);
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);

    status = milvus::ParseTargetSizeMB("1,5GB", target_size_mb, normalized);
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);

    status = milvus::ParseTargetSizeMB("abc", target_size_mb, normalized);
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
}
