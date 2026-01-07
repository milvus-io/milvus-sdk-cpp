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

#include <set>

#include "./CollectionRequestBase.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::LoadCollection()
 */
class LoadCollectionRequest : public CollectionRequestBase<LoadCollectionRequest> {
 public:
    /**
     * @brief Constructor
     */
    LoadCollectionRequest() = default;

    /**
     * @brief Sync mode.
     */
    bool
    Sync() const;

    /**
     * @brief Set sync mode. Default value is true.
     * True: wait the collection to be fully loaded.
     * False: return immediately no matter the collection is fully loaded or not.
     */
    void
    SetSync(bool sync);

    /**
     * @brief Set sync mode. Default value is true.
     * True: wait the collection to be fully loaded.
     * False: return immediately no matter the collection is fully loaded or not.
     */
    LoadCollectionRequest&
    WithSync(bool sync);

    /**
     * @brief Number of replicas.
     */
    int64_t
    ReplicaNum() const;

    /**
     * @brief Set number of replicas.
     */
    void
    SetReplicaNum(int64_t replica_num);

    /**
     * @brief Set number of replicas.
     */
    LoadCollectionRequest&
    WithReplicaNum(int64_t replica_num);

    /**
     * @brief Timeout in milliseconds.
     */
    int64_t
    TimeoutMs() const;

    /**
     * @brief Set timeout in milliseconds. Default value is 60000ms. Only work when Sync() is true.
     * If the WaitFlushedMs is zero, the LoadCollection() will call GetLoadingProgress() to loading state,
     * until the collection is fully loaded into memory.
     * If the WaitFlushedMs is larger than zero, the LoadCollection() will break the loop after a certain of time span
     * and return a status saying the process is timeout.
     */
    void
    SetTimeoutMs(int64_t timeout_ms);

    /**
     * @brief Set timeout in milliseconds. Default value is 60000ms. Only work when Sync() is true.
     * If the WaitFlushedMs is zero, the LoadCollection() will call GetLoadingProgress() to loading state,
     * until the collection is fully loaded into memory.
     * If the WaitFlushedMs is larger than zero, the LoadCollection() will break the loop after a certain of time span
     * and return a status saying the process is timeout.
     */
    LoadCollectionRequest&
    WithTimeoutMs(int64_t timeout_ms);

    /**
     * @brief Refresh option.
     */
    bool
    Refresh() const;

    /**
     * @brief Set refresh option.
     * Take effect when there are new segments generaged by bulkimport interface.
     * True: load new segments generaged by bulkimport interface.
     * False: ignore new segments generaged by bulkimport interface.
     */
    void
    SetRefresh(bool refresh);

    /**
     * @brief Set refresh option.
     * Take effect when there are new segments generaged by bulkimport interface.
     * True: load new segments generaged by bulkimport interface.
     * False: ignore new segments generaged by bulkimport interface.
     */
    LoadCollectionRequest&
    WithRefresh(bool refresh);

    /**
     * @brief Load fields.
     */
    const std::set<std::string>&
    LoadFields() const;

    /**
     * @brief Set load fields.
     */
    void
    SetLoadFields(const std::set<std::string>& load_fields);

    /**
     * @brief Set load fields.
     */
    LoadCollectionRequest&
    WithLoadFields(const std::set<std::string>& load_fields);

    /**
     * @brief Add a field to be loaded.
     */
    LoadCollectionRequest&
    AddLoadField(const std::string& field_name);

    /**
     * @brief Skip dynamic field option.
     */
    bool
    SkipDynamicField() const;

    /**
     * @brief Set skip dynamic field option.
     */
    void
    SetSkipDynamicField(bool skip_dynamic_field);

    /**
     * @brief Set skip dynamic field option.
     */
    LoadCollectionRequest&
    WithSkipDynamicField(bool skip_dynamic_field);

    /**
     * @brief Target resource groups.
     */
    const std::set<std::string>&
    TargetResourceGroups() const;

    /**
     * @brief Set target resource groups.
     */
    void
    SetTargetResourceGroups(const std::set<std::string>& target_resource_groups);

    /**
     * @brief Set target resource groups.
     */
    LoadCollectionRequest&
    WithTargetResourceGroups(const std::set<std::string>& target_resource_groups);

 private:
    bool sync_{true};
    int64_t replica_num_{1};
    int64_t timeout_ms_{60000};
    bool refresh_{false};
    std::set<std::string> load_feilds_;
    bool skip_dynamic_field_{false};
    std::set<std::string> target_resource_groups_;
};

}  // namespace milvus
