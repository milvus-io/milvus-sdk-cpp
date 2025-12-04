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

#include "milvus/request/resourcegroup/TransferNodeRequest.h"

namespace milvus {

const std::string&
TransferNodeRequest::SourceGroup() const {
    return source_group_;
}

void
TransferNodeRequest::SetSourceGroup(const std::string& source_group) {
    source_group_ = source_group;
}

TransferNodeRequest&
TransferNodeRequest::WithSourceGroup(const std::string& source_group) {
    SetSourceGroup(source_group);
    return *this;
}

const std::string&
TransferNodeRequest::TargetGroup() const {
    return target_group_;
}

void
TransferNodeRequest::SetTargetGroup(const std::string& target_group) {
    target_group_ = target_group;
}

TransferNodeRequest&
TransferNodeRequest::WithTargetGroup(const std::string& target_group) {
    SetTargetGroup(target_group);
    return *this;
}

int64_t
TransferNodeRequest::NumNodes() const {
    return num_nodes_;
}

void
TransferNodeRequest::SetNumNodes(int64_t num_nodes) {
    num_nodes_ = num_nodes;
}

TransferNodeRequest&
TransferNodeRequest::WithNumNodes(int64_t num_nodes) {
    SetNumNodes(num_nodes);
    return *this;
}

}  // namespace milvus
