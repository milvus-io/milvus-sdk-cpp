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

#include "../../types/PartitionInfo.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::ListPartitions()
 */
class ListPartitionsResponse {
 public:
    /**
     * @brief Constructor
     */
    ListPartitionsResponse() = default;

    // Getter and Setter for partition_names_
    const std::vector<std::string>&
    PartitionsNames() const;
    void
    SetPartitionNames(std::vector<std::string>&& names);

    // Getter and Setter for partition_infos_
    const std::vector<PartitionInfo>&
    PartitionInfos() const;
    void
    SetPartitionInfos(std::vector<PartitionInfo>&& infos);

 private:
    std::vector<std::string> partition_names_;
    std::vector<PartitionInfo> partition_infos_;
};

}  // namespace milvus
