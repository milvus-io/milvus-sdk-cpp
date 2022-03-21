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

#include "milvus/types/CollectionInfo.h"

namespace milvus {

CollectionInfo::CollectionInfo() = default;

CollectionInfo::CollectionInfo(std::string collection_name, int64_t collection_id, uint64_t create_time,
                               uint64_t load_percentage)
    : name_(std::move(collection_name)),
      collection_id_(collection_id),
      created_utc_timestamp_(create_time),
      in_memory_percentage_(load_percentage) {
}

const std::string&
CollectionInfo::Name() const {
    return name_;
}

int64_t
CollectionInfo::ID() const {
    return collection_id_;
}

uint64_t
CollectionInfo::CreatedTime() const {
    return created_utc_timestamp_;
}

uint64_t
CollectionInfo::MemoryPercentage() const {
    return in_memory_percentage_;
}

}  // namespace milvus
