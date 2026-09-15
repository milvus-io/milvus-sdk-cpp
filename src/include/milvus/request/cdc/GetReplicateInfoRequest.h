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

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::GetReplicateInfo()
 */
class MILVUS_SDK_API GetReplicateInfoRequest {
    /**
     * @brief Get the source cluster identifier.
     * @return the source cluster ID.
     */
 public:
    const std::string&
    SourceClusterID() const;

    /**
     * @brief Set the source cluster identifier.
     *
     * @param [in] source_cluster_id
     */
    void
    SetSourceClusterID(const std::string& source_cluster_id);

    /**
     * @brief Set the source cluster identifier.
     *
     * @param [in] source_cluster_id
     */
    GetReplicateInfoRequest&
    WithSourceClusterID(const std::string& source_cluster_id);

    /**
     * @brief Get the target Pulsar channel name.
     * @return the target p channel.
     */
    const std::string&
    TargetPChannel() const;

    /**
     * @brief Set the target Pulsar channel name.
     *
     * @param [in] target_pchannel
     */
    void
    SetTargetPChannel(const std::string& target_pchannel);

    /**
     * @brief Set the target Pulsar channel name.
     *
     * @param [in] target_pchannel
     */
    GetReplicateInfoRequest&
    WithTargetPChannel(const std::string& target_pchannel);

 private:
    std::string source_cluster_id_;
    std::string target_pchannel_;
};

}  // namespace milvus
