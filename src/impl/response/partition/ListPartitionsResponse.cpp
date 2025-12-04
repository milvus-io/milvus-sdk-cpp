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

#include "milvus/response/partition/ListPartitionsResponse.h"

#include <memory>

namespace milvus {

const std::vector<std::string>&
ListPartitionsResponse::PartitionsNames() const {
    return partition_names_;
}

void
ListPartitionsResponse::SetPartitionNames(std::vector<std::string>&& names) {
    partition_names_ = std::move(names);
}

const std::vector<PartitionInfo>&
ListPartitionsResponse::PartitionInfos() const {
    return partition_infos_;
}

void
ListPartitionsResponse::SetPartitionInfos(std::vector<PartitionInfo>&& infos) {
    partition_infos_ = std::move(infos);
}

}  // namespace milvus
