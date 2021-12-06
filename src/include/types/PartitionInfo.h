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

#include <string>
#include <vector>

namespace milvus {

/**
 * @brief Partition runtime information including create timestamp and loading percentage, returned by ShowPartitions().
 */
class PartitionInfo {
 public:
    std::string
    Name() const {
        return name_;
    }

    int64_t
    Id() const {
        return id_;
    }

    uint64_t
    CreatedUtcTimestamp() const {
        return created_utc_timestamp_;
    }

    int64_t
    InMemoryPercentage() const {
        return in_memory_percentage_;
    }

    PartitionInfo(std::string name, int64_t id, uint64_t created_utc_timestamp = 0, int64_t in_memory_percentage = 0)
        : name_(std::move(name)),
          id_(id),
          created_utc_timestamp_(created_utc_timestamp),
          in_memory_percentage_(in_memory_percentage) {
    }

 private:
    /**
     * @brief Name of this partition.
     */
    std::string name_;

    /**
     * @brief Internal id of this partition.
     */
    int64_t id_;

    /**
     * @brief The utc timestamp calculated by created_timestamp.
     */
    uint64_t created_utc_timestamp_ = 0;

    /**
     * @brief Partition loading percentage.
     */
    int64_t in_memory_percentage_ = 0;
};

inline bool
operator==(const PartitionInfo& a, const PartitionInfo& b) {
    return a.Name() == b.Name() && a.Id() && b.Id() && a.CreatedUtcTimestamp() == b.CreatedUtcTimestamp() &&
           a.InMemoryPercentage() == b.InMemoryPercentage();
}

using PartitionsInfo = std::vector<PartitionInfo>;

}  // namespace milvus
