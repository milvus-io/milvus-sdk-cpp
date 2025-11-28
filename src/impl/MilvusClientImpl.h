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

#include <map>
#include <memory>
#include <mutex>

#include "milvus/MilvusClient.h"
#include "utils/ConnectionHandler.h"

/**
 *  @brief namespace milvus
 */
namespace milvus {

class MilvusClientImpl : public MilvusClient {
 public:
    MilvusClientImpl() = default;
    ~MilvusClientImpl() override;

    Status
    Connect(const ConnectParam& param) final;

    Status
    Disconnect() final;

    Status
    SetRpcDeadlineMs(uint64_t timeout_ms) final;

    Status
    SetRetryParam(const RetryParam& retry_param) final;

    Status
    GetVersion(std::string& version) final;

    Status
    GetServerVersion(std::string& version) final;

    Status
    GetSDKVersion(std::string& version) final;

    Status
    CreateCollection(const CollectionSchema& schema, int64_t num_partitions) final;

    Status
    HasCollection(const std::string& collection_name, bool& has) final;

    Status
    DropCollection(const std::string& collection_name) final;

    Status
    LoadCollection(const std::string& collection_name, int replica_number,
                   const ProgressMonitor& progress_monitor) final;

    Status
    ReleaseCollection(const std::string& collection_name) final;

    Status
    DescribeCollection(const std::string& collection_name, CollectionDesc& collection_desc) final;

    Status
    RenameCollection(const std::string& collection_name, const std::string& new_collection_name) final;

    Status
    GetCollectionStatistics(const std::string& collection_name, CollectionStat& collection_stat,
                            const ProgressMonitor& progress_monitor) final;

    Status
    ShowCollections(const std::vector<std::string>& collection_names, CollectionsInfo& collections_info) final;

    Status
    ListCollections(CollectionsInfo& collections_info, bool only_show_loaded) final;

    Status
    GetLoadState(const std::string& collection_name, bool& is_loaded,
                 const std::vector<std::string> partition_names) final;

    Status
    AlterCollectionProperties(const std::string& collection_name,
                              const std::unordered_map<std::string, std::string>& properties) final;

    Status
    DropCollectionProperties(const std::string& collection_name, const std::set<std::string>& property_keys) final;

    Status
    AlterCollectionField(const std::string& collection_name, const std::string& field_name,
                         const std::unordered_map<std::string, std::string>& properties) final;
    Status
    CreatePartition(const std::string& collection_name, const std::string& partition_name) final;

    Status
    DropPartition(const std::string& collection_name, const std::string& partition_name) final;

    Status
    HasPartition(const std::string& collection_name, const std::string& partition_name, bool& has) final;

    Status
    LoadPartitions(const std::string& collection_name, const std::vector<std::string>& partition_names,
                   int replica_number, const ProgressMonitor& progress_monitor) final;

    Status
    ReleasePartitions(const std::string& collection_name, const std::vector<std::string>& partition_names) final;

    Status
    GetPartitionStatistics(const std::string& collection_name, const std::string& partition_name,
                           PartitionStat& partition_stat, const ProgressMonitor& progress_monitor) final;

    Status
    ShowPartitions(const std::string& collection_name, const std::vector<std::string>& partition_names,
                   PartitionsInfo& partitions_info) final;

    Status
    ListPartitions(const std::string& collection_name, PartitionsInfo& partitions_info, bool only_show_loaded) final;

    Status
    CreateAlias(const std::string& collection_name, const std::string& alias) final;

    Status
    DropAlias(const std::string& alias) final;

    Status
    AlterAlias(const std::string& collection_name, const std::string& alias) final;

    Status
    DescribeAlias(const std::string& alias_name, AliasDesc& desc) final;

    Status
    ListAliases(const std::string& collection_name, std::vector<AliasDesc>& descs) final;

    Status
    UseDatabase(const std::string& db_name) final;

    Status
    CurrentUsedDatabase(std::string& db_name) final;

    Status
    CreateDatabase(const std::string& db_name, const std::unordered_map<std::string, std::string>& properties) final;

    Status
    DropDatabase(const std::string& db_name) final;

    Status
    ListDatabases(std::vector<std::string>& names) final;

    Status
    AlterDatabaseProperties(const std::string& db_name,
                            const std::unordered_map<std::string, std::string>& properties) final;

    Status
    DropDatabaseProperties(const std::string& db_name, const std::vector<std::string>& properties) final;

    Status
    DescribeDatabase(const std::string& db_name, DatabaseDesc& db_desc) final;

    Status
    CreateIndex(const std::string& collection_name, const IndexDesc& index_desc,
                const ProgressMonitor& progress_monitor) final;

    Status
    DescribeIndex(const std::string& collection_name, const std::string& field_name, IndexDesc& index_desc) final;

    Status
    ListIndexes(const std::string& collection_name, const std::string& field_name,
                std::vector<std::string>& index_names) final;

    Status
    GetIndexState(const std::string& collection_name, const std::string& field_name, IndexState& state) final;

    Status
    GetIndexBuildProgress(const std::string& collection_name, const std::string& field_name,
                          IndexProgress& progress) final;

    Status
    DropIndex(const std::string& collection_name, const std::string& field_name) final;

    Status
    AlterIndexProperties(const std::string& collection_name, const std::string& index_name,
                         const std::unordered_map<std::string, std::string>& properties) final;

    Status
    DropIndexProperties(const std::string& collection_name, const std::string& index_name,
                        const std::set<std::string>& property_keys) final;

    Status
    Insert(const std::string& collection_name, const std::string& partition_name,
           const std::vector<FieldDataPtr>& fields, DmlResults& results) final;

    Status
    Insert(const std::string& collection_name, const std::string& partition_name, const EntityRows& rows,
           DmlResults& results) final;

    Status
    Upsert(const std::string& collection_name, const std::string& partition_name,
           const std::vector<FieldDataPtr>& fields, DmlResults& results) final;

    Status
    Upsert(const std::string& collection_name, const std::string& partition_name, const EntityRows& rows,
           DmlResults& results) final;

    Status
    Delete(const std::string& collection_name, const std::string& partition_name, const std::string& expression,
           DmlResults& results) final;

    Status
    Search(const SearchArguments& arguments, SearchResults& results) final;

    Status
    SearchIterator(SearchIteratorArguments& arguments, SearchIteratorPtr& iterator) final;

    Status
    HybridSearch(const HybridSearchArguments& arguments, SearchResults& results) final;

    Status
    Query(const QueryArguments& arguments, QueryResults& results) final;

    Status
    QueryIterator(QueryIteratorArguments& arguments, QueryIteratorPtr& iterator) final;

    Status
    RunAnalyzer(const RunAnalyzerArguments& arguments, AnalyzerResults& results) final;

    Status
    Flush(const std::vector<std::string>& collection_names, const ProgressMonitor& progress_monitor) final;

    Status
    GetFlushState(const std::vector<int64_t>& segments, bool& flushed) final;

    Status
    GetPersistentSegmentInfo(const std::string& collection_name, SegmentsInfo& segments_info) final;

    Status
    GetQuerySegmentInfo(const std::string& collection_name, QuerySegmentsInfo& segments_info) final;

    Status
    GetMetrics(const std::string& request, std::string& response, std::string& component_name) final;

    Status
    LoadBalance(int64_t src_node, const std::vector<int64_t>& dst_nodes, const std::vector<int64_t>& segments) final;

    Status
    GetCompactionState(int64_t compaction_id, CompactionState& compaction_state) final;

    Status
    ManualCompaction(const std::string& collection_name, uint64_t travel_timestamp, int64_t& compaction_id) final;

    Status
    GetCompactionPlans(int64_t compaction_id, CompactionPlans& plans) final;

    Status
    CreateCredential(const std::string& username, const std::string& password) final;

    Status
    UpdateCredential(const std::string& username, const std::string& old_password,
                     const std::string& new_password) final;

    Status
    DeleteCredential(const std::string& username) final;

    Status
    ListCredUsers(std::vector<std::string>& users) final;

    Status
    CreateResourceGroup(const std::string& name, const ResourceGroupConfig& config) final;

    Status
    DropResourceGroup(const std::string& name) final;

    Status
    UpdateResourceGroups(const std::unordered_map<std::string, ResourceGroupConfig>& groups) final;

    Status
    TransferNode(const std::string& source_group, const std::string& target_group, uint32_t num_nodes) final;

    Status
    TransferReplica(const std::string& source_group, const std::string& target_group,
                    const std::string& collection_name, uint32_t num_replicas) final;

    Status
    ListResourceGroups(std::vector<std::string>& group_names) final;

    Status
    DescribeResourceGroup(const std::string& group_name, ResourceGroupDesc& desc) final;

    Status
    CreateUser(const std::string& user_name, const std::string& password) final;

    Status
    UpdatePassword(const std::string& user_name, const std::string& old_password,
                   const std::string& new_password) final;

    Status
    DropUser(const std::string& user_name) final;

    Status
    DescribeUser(const std::string& user_name, UserDesc& desc) final;

    Status
    ListUsers(std::vector<std::string>& names) final;

    Status
    CreateRole(const std::string& role_name) final;

    Status
    DropRole(const std::string& role_name, bool force_drop) final;

    Status
    DescribeRole(const std::string& role_name, RoleDesc& desc) final;

    Status
    ListRoles(std::vector<std::string>& names) final;

    Status
    GrantRole(const std::string& user_name, const std::string& role_name) final;

    Status
    RevokeRole(const std::string& user_name, const std::string& role_name) final;

    Status
    GrantPrivilege(const std::string& role_name, const std::string& privilege, const std::string& collection_name,
                   const std::string& db_name) final;

    Status
    RevokePrivilege(const std::string& role_name, const std::string& privilege, const std::string& collection_name,
                    const std::string& db_name) final;

    Status
    CreatePrivilegeGroup(const std::string& group_name) final;

    Status
    DropPrivilegeGroup(const std::string& group_name) final;

    Status
    ListPrivilegeGroups(PrivilegeGroupInfos& groups) final;

    Status
    AddPrivilegesToGroup(const std::string& group_name, const std::vector<std::string>& privileges) final;

    Status
    RemovePrivilegesFromGroup(const std::string& group_name, const std::vector<std::string>& privileges) final;

 private:
    /**
     * @brief return desc if it is existing, else call describeCollection() and cache it
     */
    Status
    getCollectionDesc(const std::string& collection_name, bool force_update, CollectionDescPtr& desc_ptr);

    /**
     * @brief clean desc of all the collections in the cache
     */
    void
    cleanCollectionDescCache();

    /**
     * @brief remove a collections's desc from the cache
     */
    void
    removeCollectionDesc(const std::string& collection_name);

    template <typename ArgClass>
    Status
    iteratorPrepare(ArgClass& arguments);

 private:
    ConnectionHandler connection_;

    // cache of collection schemas
    // this cache is db level, once useDatabase() is called, this cache will be cleaned
    // so, it is fine to use collection name as key, no need to involve db name
    std::map<std::string, CollectionDescPtr> collection_desc_cache_;
    std::mutex collection_desc_cache_mtx_;
};

}  // namespace milvus
