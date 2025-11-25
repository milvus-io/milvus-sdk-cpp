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

namespace milvus {

/**
 * @brief Used by MilvusClientV2::TransferNode()
 */
class TransferNodeRequest {
 public:
    /**
     * @brief Constructor
     */
    TransferNodeRequest() = default;

    /**
     * @brief Get name of the source resource group.
     */
    const std::string&
    SourceGroup() const;

    /**
     * @brief Set name of the source resource group.
     */
    void
    SetSourceGroup(const std::string& source_group);

    /**
     * @brief Set name of the source resource group.
     */
    TransferNodeRequest&
    WithSourceGroup(const std::string& source_group);

    /**
     *  Get name of the target resource group.
     */
    const std::string&
    TargetGroup() const;

    /**
     *  Set name of the target resource group.
     */
    void
    SetTargetGroup(const std::string& target_group);

    /**
     *  Set name of the target resource group.
     */
    TransferNodeRequest&
    WithTargetGroup(const std::string& target_group);

    /**
     *  Get number of nodes to transfer.
     */
    int64_t
    NumNodes() const;

    /**
     *  Set number of nodes to transfer.
     */
    void
    SetNumNodes(int64_t num_nodes);

    /**
     *  Set number of nodes to transfer.
     */
    TransferNodeRequest&
    WithNumNodes(int64_t num_nodes);

 private:
    std::string source_group_;
    std::string target_group_;
    int64_t num_nodes_;
};

}  // namespace milvus
