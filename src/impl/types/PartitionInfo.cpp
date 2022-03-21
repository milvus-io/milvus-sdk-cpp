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

#include "milvus/types/PartitionInfo.h"

namespace milvus {
PartitionInfo::PartitionInfo(std::string name, int64_t id, uint64_t created_utc_timestamp, int64_t in_memory_percentage)
    : name_(std::move(name)),
      id_(id),
      created_utc_timestamp_(created_utc_timestamp),
      in_memory_percentage_(in_memory_percentage) {
}

std::string
PartitionInfo::Name() const {
    return name_;
}

int64_t
PartitionInfo::Id() const {
    return id_;
}

uint64_t
PartitionInfo::CreatedUtcTimestamp() const {
    return created_utc_timestamp_;
}

int64_t
PartitionInfo::InMemoryPercentage() const {
    return in_memory_percentage_;
}

bool
PartitionInfo::Loaded() const {
    return in_memory_percentage_ >= 100;
}

bool
operator==(const PartitionInfo& a, const PartitionInfo& b) {
    return a.Name() == b.Name() && a.Id() && b.Id() && a.CreatedUtcTimestamp() == b.CreatedUtcTimestamp() &&
           a.InMemoryPercentage() == b.InMemoryPercentage();
}

}  // namespace milvus
