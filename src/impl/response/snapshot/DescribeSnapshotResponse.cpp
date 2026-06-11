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

#include "milvus/response/snapshot/DescribeSnapshotResponse.h"

namespace milvus {

const std::string&
DescribeSnapshotResponse::Name() const {
    return name_;
}

void
DescribeSnapshotResponse::SetName(std::string name) {
    name_ = std::move(name);
}

const std::string&
DescribeSnapshotResponse::Description() const {
    return description_;
}

void
DescribeSnapshotResponse::SetDescription(std::string description) {
    description_ = std::move(description);
}

const std::string&
DescribeSnapshotResponse::CollectionName() const {
    return collection_name_;
}

void
DescribeSnapshotResponse::SetCollectionName(std::string collection_name) {
    collection_name_ = std::move(collection_name);
}

const std::vector<std::string>&
DescribeSnapshotResponse::PartitionNames() const {
    return partition_names_;
}

void
DescribeSnapshotResponse::SetPartitionNames(std::vector<std::string>&& partition_names) {
    partition_names_ = std::move(partition_names);
}

int64_t
DescribeSnapshotResponse::CreateTs() const {
    return create_ts_;
}

void
DescribeSnapshotResponse::SetCreateTs(int64_t create_ts) {
    create_ts_ = create_ts;
}

const std::string&
DescribeSnapshotResponse::S3Location() const {
    return s3_location_;
}

void
DescribeSnapshotResponse::SetS3Location(std::string s3_location) {
    s3_location_ = std::move(s3_location);
}

}  // namespace milvus
