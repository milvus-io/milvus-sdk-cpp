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

#include "milvus/types/PartitionStat.h"

namespace milvus {
uint64_t
PartitionStat::RowCount() const {
    const auto iter = statistics_.find(KeyRowCount());
    if (iter == statistics_.end()) {
        // TODO: throw exception or log
        return 0;
    }
    return std::strtoll(iter->second.c_str(), nullptr, 10);
}

void
PartitionStat::SetName(std::string name) {
    name_ = std::move(name);
}

const std::string&
PartitionStat::Name() const {
    return name_;
}

void
PartitionStat::Emplace(std::string key, std::string value) {
    statistics_.emplace(std::move(key), std::move(value));
}

}  // namespace milvus
