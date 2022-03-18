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

#include "milvus/types/CollectionDesc.h"

#include "milvus/types/CollectionSchema.h"

namespace milvus {

struct CollectionDesc::Impl {
    CollectionSchema schema_;

    int64_t collection_id_;
    std::vector<std::string> alias_;
    uint64_t created_utc_timestamp_ = 0;
};

CollectionDesc::CollectionDesc() : impl_(new Impl) {
}

CollectionDesc::CollectionDesc(CollectionDesc&&) noexcept = default;

CollectionDesc::~CollectionDesc() = default;

const CollectionSchema&
CollectionDesc::Schema() const {
    return impl_->schema_;
}

void
CollectionDesc::SetSchema(const CollectionSchema& schema) {
    impl_->schema_ = schema;
}

int64_t
CollectionDesc::ID() const {
    return impl_->collection_id_;
}

void
CollectionDesc::SetID(const int64_t id) {
    impl_->collection_id_ = id;
}

const std::vector<std::string>&
CollectionDesc::Alias() const {
    return impl_->alias_;
}

void
CollectionDesc::SetAlias(const std::vector<std::string>& alias) {
    impl_->alias_ = alias;
}

uint64_t
CollectionDesc::CreatedTime() const {
    return impl_->created_utc_timestamp_;
}

void
CollectionDesc::SetCreatedTime(const uint64_t ts) {
    impl_->created_utc_timestamp_ = ts;
}

}  // namespace milvus
