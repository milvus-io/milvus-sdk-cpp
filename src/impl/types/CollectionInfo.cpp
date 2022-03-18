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
struct CollectionInfo::Impl {
    std::string name_;
    int64_t collection_id_ = 0;
    uint64_t created_utc_timestamp_ = 0;
    uint64_t in_memory_percentage_ = 0;

    Impl() = default;

    Impl(const std::string& name, int64_t collection_id, uint64_t create_time, uint64_t load_percentage)
        : name_(name),
          collection_id_(collection_id),
          created_utc_timestamp_(create_time),
          in_memory_percentage_(load_percentage) {
    }
};

CollectionInfo::CollectionInfo() : impl_(new Impl) {
}

CollectionInfo::CollectionInfo(const std::string& collection_name, int64_t collection_id, uint64_t create_time,
                               uint64_t load_percentage)
    : impl_(new Impl(collection_name, collection_id, create_time, load_percentage)) {
}

CollectionInfo::CollectionInfo(CollectionInfo&&) noexcept = default;

CollectionInfo::~CollectionInfo() = default;

const std::string&
CollectionInfo::Name() const {
    return impl_->name_;
}

int64_t
CollectionInfo::ID() const {
    return impl_->collection_id_;
}

uint64_t
CollectionInfo::CreatedTime() const {
    return impl_->created_utc_timestamp_;
}

uint64_t
CollectionInfo::MemoryPercentage() const {
    return impl_->in_memory_percentage_;
}

}  // namespace milvus
