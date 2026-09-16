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

#include "milvus/Export.h"
#include "milvus/types/ReplicateConfiguration.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::UpdateReplicateConfiguration()
 */
class MILVUS_SDK_API UpdateReplicateConfigurationRequest {
    /**
     * @brief Get the replication configuration.
     * @return the configuration.
     */
 public:
    const ReplicateConfiguration&
    Configuration() const;

    /**
     * @brief Set the replication configuration.
     *
     * @param [in] configuration
     */
    void
    SetConfiguration(ReplicateConfiguration&& configuration);

    /**
     * @brief Set the replication configuration.
     *
     * @param [in] configuration
     */
    UpdateReplicateConfigurationRequest&
    WithConfiguration(ReplicateConfiguration&& configuration);

    /**
     * @brief Get whether to force-promote the configuration.
     * @return the force promote.
     */
    bool
    ForcePromote() const;

    /**
     * @brief Set whether to force-promote the configuration.
     *
     * @param [in] force_promote
     */
    void
    SetForcePromote(bool force_promote);

    /**
     * @brief Set whether to force-promote the configuration.
     *
     * @param [in] force_promote
     */
    UpdateReplicateConfigurationRequest&
    WithForcePromote(bool force_promote);

 private:
    ReplicateConfiguration configuration_;
    bool force_promote_{false};
};

}  // namespace milvus
