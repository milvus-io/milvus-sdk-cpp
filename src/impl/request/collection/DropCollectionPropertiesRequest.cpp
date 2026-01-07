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

#include "milvus/request/collection/DropCollectionPropertiesRequest.h"

#include <memory>

namespace milvus {

const std::set<std::string>&
DropCollectionPropertiesRequest::PropertyKeys() const {
    return property_keys_;
}

void
DropCollectionPropertiesRequest::SetPropertyKeys(std::set<std::string>&& keys) {
    property_keys_ = std::move(keys);
}

DropCollectionPropertiesRequest&
DropCollectionPropertiesRequest::WithPropertyKeys(std::set<std::string>&& keys) {
    SetPropertyKeys(std::move(keys));
    return *this;
}

DropCollectionPropertiesRequest&
DropCollectionPropertiesRequest::AddPropertyKey(const std::string& key) {
    property_keys_.insert(key);
    return *this;
}

}  // namespace milvus
