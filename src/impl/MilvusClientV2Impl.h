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

#include "milvus/MilvusClientV2.h"
#include "utils/ConnectionHandler.h"

namespace milvus {

class MilvusClientV2Impl : public MilvusClientV2 {
 public:
    MilvusClientV2Impl() = default;
    ~MilvusClientV2Impl() override;

    Status
    Connect(const ConnectParam& param) final;

    Status
    Disconnect() final;

    Status
    SetRpcDeadlineMs(uint64_t timeout_ms) final;

    Status
    SetRetryParam(const RetryParam& retry_param) final;

    Status
    GetServerVersion(std::string& version) final;

    Status
    GetSDKVersion(std::string& version) final;

    Status
    CheckHealth(const CheckHealthRequest& request, CheckHealthResponse& response) final;

    Status
    CreateCollection(const CreateCollectionRequest& request) final;

    Status
    HasCollection(const HasCollectionRequest& request, HasCollectionResponse& response) final;

    Status
    DropCollection(const DropCollectionRequest& request) final;

    Status
    LoadCollection(const LoadCollectionRequest& request) final;

    Status
    ReleaseCollection(const ReleaseCollectionRequest& request) final;

    Status
    DescribeCollection(const DescribeCollectionRequest& request, DescribeCollectionResponse& response) final;

    Status
    RenameCollection(const RenameCollectionRequest& request) final;

    Status
    GetCollectionStats(const GetCollectionStatsRequest& request, GetCollectionStatsResponse& response) final;

    Status
    ListCollections(const ListCollectionsRequest& request, ListCollectionsResponse& response) final;

    Status
    GetLoadState(const GetLoadStateRequest& request, GetLoadStateResponse& response) final;

    Status
    AlterCollectionProperties(const AlterCollectionPropertiesRequest& request) final;

    Status
    DropCollectionProperties(const DropCollectionPropertiesRequest& request) final;

    Status
    AlterCollectionFieldProperties(const AlterCollectionFieldPropertiesRequest& request) final;

    Status
    DropCollectionFieldProperties(const DropCollectionFieldPropertiesRequest& request) final;

    Status
    CreatePartition(const CreatePartitionRequest& request) final;

    Status
    DropPartition(const DropPartitionRequest& request) final;

    Status
    HasPartition(const HasPartitionRequest& request, HasPartitionResponse& response) final;

    Status
    LoadPartitions(const LoadPartitionsRequest& request) final;

    Status
    ReleasePartitions(const ReleasePartitionsRequest& request) final;

    Status
    GetPartitionStatistics(const GetPartitionStatsRequest& request, GetPartitionStatsResponse& response) final;

    Status
    ListPartitions(const ListPartitionsRequest& request, ListPartitionsResponse& response) final;

    Status
    CreateAlias(const CreateAliasRequest& request) final;

    Status
    DropAlias(const DropAliasRequest& request) final;

    Status
    AlterAlias(const AlterAliasRequest& request) final;

    Status
    DescribeAlias(const DescribeAliasRequest& request, DescribeAliasResponse& response) final;

    Status
    ListAliases(const ListAliasesRequest& request, ListAliasesResponse& response) final;

    Status
    UseDatabase(const std::string& db_name) final;

    Status
    CurrentUsedDatabase(std::string& db_name) final;

    Status
    CreateDatabase(const CreateDatabaseRequest& request) final;

    Status
    DropDatabase(const DropDatabaseRequest& request) final;

    Status
    ListDatabases(const ListDatabasesRequest& request, ListDatabasesResponse& response) final;

    Status
    AlterDatabaseProperties(const AlterDatabasePropertiesRequest& request) final;

    Status
    DropDatabaseProperties(const DropDatabasePropertiesRequest& request) final;

    Status
    DescribeDatabase(const DescribeDatabaseRequest& request, DescribeDatabaseResponse& response) final;

    Status
    CreateIndex(const CreateIndexRequest& request) final;

    Status
    DescribeIndex(const DescribeIndexRequest& request, DescribeIndexResponse& response) final;

    Status
    ListIndexes(const ListIndexesRequest& request, ListIndexesResponse& response) final;

    Status
    DropIndex(const DropIndexRequest& request) final;

    Status
    AlterIndexProperties(const AlterIndexPropertiesRequest& request) final;

    Status
    DropIndexProperties(const DropIndexPropertiesRequest& request) final;

    Status
    Insert(const InsertRequest& request, InsertResponse& response) final;

    Status
    Upsert(const UpsertRequest& request, UpsertResponse& response) final;

    Status
    Delete(const DeleteRequest& request, DeleteResponse& response) final;

    Status
    Search(const SearchRequest& request, SearchResponse& response) final;

    Status
    SearchIterator(SearchIteratorRequest& request, SearchIteratorPtr& response) final;

    Status
    HybridSearch(const HybridSearchRequest& request, HybridSearchResponse& response) final;

    Status
    Query(const QueryRequest& request, QueryResponse& response) final;

    Status
    QueryIterator(QueryIteratorRequest& request, QueryIteratorPtr& response) final;

    Status
    RunAnalyzer(const RunAnalyzerRequest& request, RunAnalyzerResponse& response) final;

    Status
    Flush(const FlushRequest& request) final;

    Status
    ListPersistentSegments(const ListPersistentSegmentsRequest& request,
                           ListPersistentSegmentsResponse& response) final;

    Status
    ListQuerySegments(const ListQuerySegmentsRequest& request, ListQuerySegmentsResponse& response) final;

    Status
    Compact(const CompactRequest& request, CompactResponse& response) final;

    Status
    GetCompactionState(const GetCompactionStateRequest& request, GetCompactionStateResponse& response) final;

    Status
    GetCompactionPlans(const GetCompactionPlansRequest& request, GetCompactionPlansResponse& response) final;

    Status
    CreateResourceGroup(const CreateResourceGroupRequest& request) final;

    Status
    DropResourceGroup(const DropResourceGroupRequest& request) final;

    Status
    UpdateResourceGroups(const UpdateResourceGroupsRequest& request) final;

    Status
    TransferNode(const TransferNodeRequest& request) final;

    Status
    TransferReplica(const TransferReplicaRequest& request) final;

    Status
    ListResourceGroups(const ListResourceGroupsRequest& request, ListResourceGroupsResponse& response) final;

    Status
    DescribeResourceGroup(const DescribeResourceGroupRequest& request, DescribeResourceGroupResponse& response) final;

    Status
    CreateUser(const CreateUserRequest& request) final;

    Status
    UpdatePassword(const UpdatePasswordRequest& request) final;

    Status
    DropUser(const DropUserRequest& request) final;

    Status
    DescribeUser(const DescribeUserRequest& request, DescribeUserResponse& response) final;

    Status
    ListUsers(const ListUsersRequest& request, ListUsersResponse& response) final;

    Status
    CreateRole(const CreateRoleRequest& request) final;

    Status
    DropRole(const DropRoleRequest& request) final;

    Status
    DescribeRole(const DescribeRoleRequest& request, DescribeRoleResponse& response) final;

    Status
    ListRoles(const ListRolesRequest& request, ListRolesResponse& response) final;

    Status
    GrantRole(const GrantRoleRequest& request) final;

    Status
    RevokeRole(const RevokeRoleRequest& request) final;

    Status
    GrantPrivilegeV2(const GrantPrivilegeV2Request& request) final;

    Status
    RevokePrivilegeV2(const RevokePrivilegeV2Request& request) final;

    Status
    CreatePrivilegeGroup(const CreatePrivilegeGroupRequest& request) final;

    Status
    DropPrivilegeGroup(const DropPrivilegeGroupRequest& request) final;

    Status
    ListPrivilegeGroups(const ListPrivilegeGroupsRequest& request, ListPrivilegeGroupsResponse& response) final;

    Status
    AddPrivilegesToGroup(const AddPrivilegesToGroupRequest& request) final;

    Status
    RemovePrivilegesFromGroup(const RemovePrivilegesFromGroupRequest& request) final;

 private:
    Status
    createIndex(const std::string& db_name, const std::string& collection_name, const IndexDesc& desc, bool sync,
                int64_t timeout_ms);

    Status
    getFlushState(const std::vector<int64_t>& segments, bool& flushed);

    Status
    getCollectionDesc(const std::string& db_name, const std::string& collection_name, bool force_update,
                      CollectionDescPtr& desc_ptr);

    void
    cleanCollectionDescCache();

    void
    removeCollectionDesc(const std::string& db_name, const std::string& collection_name);

    template <typename RequestClass>
    Status
    iteratorPrepare(RequestClass& request);

 private:
    ConnectionHandler connection_;

    // cache of collection schemas
    // this cache is db level, once useDatabase() is called, this cache will be cleaned
    // so, it is fine to use collection name as key, no need to involve db name
    std::map<std::string, CollectionDescPtr> collection_desc_cache_;
    std::mutex collection_desc_cache_mtx_;
};

}  // namespace milvus
