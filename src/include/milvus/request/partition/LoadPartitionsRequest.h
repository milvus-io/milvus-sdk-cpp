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
#include <string>

namespace milvus {

/**
 * @brief Used by MilvusClientV2::LoadPartitions()
 */
class LoadPartitionsRequest {
 public:
    /**
     * @brief Constructor
     */
    LoadPartitionsRequest() = default;

    /**
     * @brief Database name in which the collection is created.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name in which the collection is created.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set database name in which the collection is created.
     */
    LoadPartitionsRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Name of the collection.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set name of the collection.
     */
    void
    SetCollectionName(const std::string& collection_name);

    /**
     * @brief Set name of the collection.
     */
    LoadPartitionsRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Name of the partitions.
     */
    const std::set<std::string>&
    PartitionNames() const;

    /**
     * @brief Set name of the partitions.
     */
    void
    SetPartitionNames(const std::set<std::string>& partition_names);

    /**
     * @brief Set new name of the partitions.
     */
    LoadPartitionsRequest&
    WithPartitionNames(const std::set<std::string>& partition_names);

    /**
     * @brief Add a partition to be loaded.
     */
    LoadPartitionsRequest&
    AddPartitionName(const std::string& partition_name);

    /**
     * @brief Get sync mode.
     */
    bool
    Sync() const;

    /**
     * @brief Set sync mode. Default value is true.
     * True: wait the partitions to be fully loaded.
     * False: return immediately no matter the partitions are fully loaded or not.
     */
    void
    SetSync(bool sync);

    /**
     * @brief Set sync mode. Default value is true.
     * True: wait the partitions to be fully loaded.
     * False: return immediately no matter the partitions are fully loaded or not.
     */
    LoadPartitionsRequest&
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
    LoadPartitionsRequest&
    WithReplicaNum(int64_t replica_num);

    /**
     * @brief Timeout in milliseconds.
     */
    int64_t
    TimeoutMs() const;

    /**
     * @brief Set timeout in milliseconds. Default value is 60000ms. Only work when Sync() is true.
     * If the WaitFlushedMs is zero, the LoadPartitions() will call GetLoadingProgress() to loading state,
     * until the collection is fully loaded into memory.
     * If the WaitFlushedMs is larger than zero, the LoadPartitions() will break the loop after a certain of time span
     * and return a status saying the process is timeout.
     *
     */
    void
    SetTimeoutMs(int64_t timeout_ms);

    /**
     * @brief Set timeout in milliseconds. Default value is 60000ms. Only work when Sync() is true.
     * If the WaitFlushedMs is zero, the LoadPartitions() will call GetLoadingProgress() to loading state,
     * until the collection is fully loaded into memory.
     * If the WaitFlushedMs is larger than zero, the LoadPartitions() will break the loop after a certain of time span
     * and return a status saying the process is timeout.
     */
    LoadPartitionsRequest&
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
    LoadPartitionsRequest&
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
    LoadPartitionsRequest&
    WithLoadFields(const std::set<std::string>& load_fields);

    /**
     * @brief Add a load field.
     */
    LoadPartitionsRequest&
    AddLoadField(const std::string& load_field);

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
    LoadPartitionsRequest&
    WithSkipDynamicField(bool skip_dynamic_field);

    /**
     * @brief Target resource groups.
     */
    const std::set<std::string>&
    TargetResourceGroups() const;

    /**
     * @brief Set target resource groups.
     * If the target_resource_groups is empty, will load into the default resource group.
     */
    void
    SetTargetResourceGroups(const std::set<std::string>& target_resource_groups);

    /**
     * @brief Set target resource groups.
     * If the target_resource_groups is empty, will load into the default resource group.
     */
    LoadPartitionsRequest&
    WithTargetResourceGroups(const std::set<std::string>& target_resource_groups);

    /**
     * @brief Add a target resource group.
     */
    LoadPartitionsRequest&
    AddTargetResourceGroups(const std::string& target_resource_group);

 private:
    std::string db_name_;
    std::string collection_name_;
    std::set<std::string> partition_names_;
    bool sync_{true};
    int64_t replica_num_{1};
    int64_t timeout_ms_{60000};
    bool refresh_{false};
    std::set<std::string> load_feilds_;
    bool skip_dynamic_field_{false};
    std::set<std::string> target_resource_groups_;
};

}  // namespace milvus
