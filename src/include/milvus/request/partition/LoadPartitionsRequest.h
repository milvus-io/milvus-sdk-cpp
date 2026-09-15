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

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::LoadPartitions()
 */
class MILVUS_SDK_API LoadPartitionsRequest {
 public:
    /**
     * @brief Constructor
     */
    LoadPartitionsRequest() = default;

    /**
     * @brief Database name in which the collection is created.
     * @return the database name.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name in which the collection is created.
     * @param [in] db_name the DB name.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set database name in which the collection is created.
     * @param [in] db_name the DB name.
     */
    LoadPartitionsRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Name of the collection.
     * @return the collection name.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set name of the collection.
     * @param [in] collection_name the collection name.
     */
    void
    SetCollectionName(const std::string& collection_name);

    /**
     * @brief Set name of the collection.
     * @param [in] collection_name the collection name.
     */
    LoadPartitionsRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Name of the partitions.
     * @return the partition names.
     */
    const std::set<std::string>&
    PartitionNames() const;

    /**
     * @brief Set name of the partitions.
     * @param [in] partition_names the partition names.
     */
    void
    SetPartitionNames(const std::set<std::string>& partition_names);

    /**
     * @brief Set new name of the partitions.
     * @param [in] partition_names the partition names.
     */
    LoadPartitionsRequest&
    WithPartitionNames(const std::set<std::string>& partition_names);

    /**
     * @brief Add a partition to be loaded.
     * @param [in] partition_name the partition name.
     */
    LoadPartitionsRequest&
    AddPartitionName(const std::string& partition_name);

    /**
     * @brief Get sync mode.
     * @return the sync.
     */
    bool
    Sync() const;

    /**
     * @brief Set sync mode. Default value is true.
     * True: wait the partitions to be fully loaded.
     * False: return immediately no matter the partitions are fully loaded or not.
     * @param [in] sync the sync.
     */
    void
    SetSync(bool sync);

    /**
     * @brief Set sync mode. Default value is true.
     * True: wait the partitions to be fully loaded.
     * False: return immediately no matter the partitions are fully loaded or not.
     * @param [in] sync the sync.
     */
    LoadPartitionsRequest&
    WithSync(bool sync);

    /**
     * @brief Number of replicas.
     * @return the replica num.
     */
    int64_t
    ReplicaNum() const;

    /**
     * @brief Set number of replicas.
     * @param [in] replica_num the replica num.
     */
    void
    SetReplicaNum(int64_t replica_num);

    /**
     * @brief Set number of replicas.
     * @param [in] replica_num the replica num.
     */
    LoadPartitionsRequest&
    WithReplicaNum(int64_t replica_num);

    /**
     * @brief Timeout in milliseconds.
     * @return the timeout ms.
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
     * @param [in] timeout_ms the timeout ms.
     */
    void
    SetTimeoutMs(int64_t timeout_ms);

    /**
     * @brief Set timeout in milliseconds. Default value is 60000ms. Only work when Sync() is true.
     * If the WaitFlushedMs is zero, the LoadPartitions() will call GetLoadingProgress() to loading state,
     * until the collection is fully loaded into memory.
     * If the WaitFlushedMs is larger than zero, the LoadPartitions() will break the loop after a certain of time span
     * and return a status saying the process is timeout.
     * @param [in] timeout_ms the timeout ms.
     */
    LoadPartitionsRequest&
    WithTimeoutMs(int64_t timeout_ms);

    /**
     * @brief Refresh option.
     * @return the refresh.
     */
    bool
    Refresh() const;

    /**
     * @brief Set refresh option.
     * Take effect when there are new segments generaged by bulkimport interface.
     * True: load new segments generaged by bulkimport interface.
     * False: ignore new segments generaged by bulkimport interface.
     * @param [in] refresh the refresh.
     */
    void
    SetRefresh(bool refresh);

    /**
     * @brief Set refresh option.
     * Take effect when there are new segments generaged by bulkimport interface.
     * True: load new segments generaged by bulkimport interface.
     * False: ignore new segments generaged by bulkimport interface.
     * @param [in] refresh the refresh.
     */
    LoadPartitionsRequest&
    WithRefresh(bool refresh);

    /**
     * @brief Load fields.
     * @return the load fields.
     */
    const std::set<std::string>&
    LoadFields() const;

    /**
     * @brief Set load fields.
     * @param [in] load_fields the load fields.
     */
    void
    SetLoadFields(const std::set<std::string>& load_fields);

    /**
     * @brief Set load fields.
     * @param [in] load_fields the load fields.
     */
    LoadPartitionsRequest&
    WithLoadFields(const std::set<std::string>& load_fields);

    /**
     * @brief Add a load field.
     * @param [in] load_field the load field.
     */
    LoadPartitionsRequest&
    AddLoadField(const std::string& load_field);

    /**
     * @brief Skip dynamic field option.
     * @return the skip dynamic field.
     */
    bool
    SkipDynamicField() const;

    /**
     * @brief Set skip dynamic field option.
     * @param [in] skip_dynamic_field the skip dynamic field.
     */
    void
    SetSkipDynamicField(bool skip_dynamic_field);

    /**
     * @brief Set skip dynamic field option.
     * @param [in] skip_dynamic_field the skip dynamic field.
     */
    LoadPartitionsRequest&
    WithSkipDynamicField(bool skip_dynamic_field);

    /**
     * @brief Target resource groups.
     * @return the target resource groups.
     */
    const std::set<std::string>&
    TargetResourceGroups() const;

    /**
     * @brief Set target resource groups.
     * If the target_resource_groups is empty, will load into the default resource group.
     * @param [in] target_resource_groups the target resource groups.
     */
    void
    SetTargetResourceGroups(const std::set<std::string>& target_resource_groups);

    /**
     * @brief Set target resource groups.
     * If the target_resource_groups is empty, will load into the default resource group.
     * @param [in] target_resource_groups the target resource groups.
     */
    LoadPartitionsRequest&
    WithTargetResourceGroups(const std::set<std::string>& target_resource_groups);

    /**
     * @brief Add a target resource group.
     * @param [in] target_resource_group the target resource group.
     */
    LoadPartitionsRequest&
    AddTargetResourceGroups(const std::string& target_resource_group);

    /**
     * @brief Load priority.
     * The load priority of the partitions. Set "low" to select low priority; any other value (including "high")
     * defaults to high priority.
     * @return the load priority.
     */
    const std::string&
    LoadPriority() const;

    /**
     * @brief Set load priority.
     * The load priority of the partitions. Set "low" to select low priority; any other value (including "high")
     * defaults to high priority.
     * @param [in] load_priority the load priority.
     */
    void
    SetLoadPriority(const std::string& load_priority);

    /**
     * @brief Set load priority.
     * The load priority of the partitions. Set "low" to select low priority; any other value (including "high")
     * defaults to high priority.
     * @param [in] load_priority the load priority.
     */
    LoadPartitionsRequest&
    WithLoadPriority(const std::string& load_priority);

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
    std::string load_priority_;
};

}  // namespace milvus
