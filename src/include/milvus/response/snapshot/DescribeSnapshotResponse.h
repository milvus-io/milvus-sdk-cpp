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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::DescribeSnapshot()
 */
class MILVUS_SDK_API DescribeSnapshotResponse {
 public:
    DescribeSnapshotResponse() = default;

    const std::string&
    Name() const;

    void
    SetName(std::string name);

    const std::string&
    Description() const;

    void
    SetDescription(std::string description);

    const std::string&
    CollectionName() const;

    void
    SetCollectionName(std::string collection_name);

    const std::vector<std::string>&
    PartitionNames() const;

    void
    SetPartitionNames(std::vector<std::string>&& partition_names);

    int64_t
    CreateTs() const;

    void
    SetCreateTs(int64_t create_ts);

    const std::string&
    S3Location() const;

    void
    SetS3Location(std::string s3_location);

 private:
    std::string name_;
    std::string description_;
    std::string collection_name_;
    std::vector<std::string> partition_names_;
    int64_t create_ts_{0};
    std::string s3_location_;
};

}  // namespace milvus
