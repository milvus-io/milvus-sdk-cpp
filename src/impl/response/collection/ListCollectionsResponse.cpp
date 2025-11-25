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

#include "milvus/response/collection/ListCollectionsResponse.h"

#include <memory>

namespace milvus {

// Getter for collection_names_
const std::vector<std::string>&
ListCollectionsResponse::CollectionNames() const {
    return collection_names_;
}

// Setter for collection_names_
void
ListCollectionsResponse::SetCollectionNames(std::vector<std::string>&& names) {
    collection_names_ = std::move(names);
}

// Getter for collection_infos_
const std::vector<CollectionInfo>&
ListCollectionsResponse::CollectionInfos() const {
    return collection_infos_;
}

// Setter for collection_infos_
void
ListCollectionsResponse::SetCollectionInfos(std::vector<CollectionInfo>&& infos) {
    collection_infos_ = std::move(infos);
}

}  // namespace milvus
