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

#include "milvus/request/utility/RefreshExternalCollectionRequest.h"

namespace milvus {

const std::string&
RefreshExternalCollectionRequest::ExternalSource() const {
    return external_source_;
}

void
RefreshExternalCollectionRequest::SetExternalSource(const std::string& external_source) {
    external_source_ = external_source;
}

RefreshExternalCollectionRequest&
RefreshExternalCollectionRequest::WithExternalSource(const std::string& external_source) {
    SetExternalSource(external_source);
    return *this;
}

const nlohmann::json&
RefreshExternalCollectionRequest::ExternalSpec() const {
    return external_spec_;
}

void
RefreshExternalCollectionRequest::SetExternalSpec(const nlohmann::json& external_spec) {
    external_spec_ = external_spec;
}

RefreshExternalCollectionRequest&
RefreshExternalCollectionRequest::WithExternalSpec(const nlohmann::json& external_spec) {
    SetExternalSpec(external_spec);
    return *this;
}

}  // namespace milvus
