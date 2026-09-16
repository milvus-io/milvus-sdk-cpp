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
 * @brief Used by MilvusClientV2::GetReplicateInfo()
 */
class MILVUS_SDK_API GetReplicateInfoResponse {
    /**
     * @brief Get the replication checkpoint.
     * @return the checkpoint.
     */
 public:
    const ReplicateCheckpoint&
    Checkpoint() const;

    /**
     * @brief Set the replication checkpoint.
     *
     * @param [in] checkpoint
     */
    void
    SetCheckpoint(ReplicateCheckpoint&& checkpoint);

    /**
     * @brief Get the salvage checkpoint.
     * @return the salvage checkpoint.
     */
    const ReplicateCheckpoint&
    SalvageCheckpoint() const;

    /**
     * @brief Set the salvage checkpoint.
     *
     * @param [in] checkpoint
     */
    void
    SetSalvageCheckpoint(ReplicateCheckpoint&& checkpoint);

 private:
    ReplicateCheckpoint checkpoint_;
    ReplicateCheckpoint salvage_checkpoint_;
};

}  // namespace milvus
