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

#include "Status.h"
#include "request/alias/AlterAliasRequest.h"
#include "request/alias/CreateAliasRequest.h"
#include "request/alias/DescribeAliasRequest.h"
#include "request/alias/DropAliasRequest.h"
#include "request/alias/ListAliasesRequest.h"
#include "request/collection/AddCollectionFieldRequest.h"
#include "request/collection/AlterCollectionFieldPropertiesRequest.h"
#include "request/collection/AlterCollectionPropertiesRequest.h"
#include "request/collection/CreateCollectionRequest.h"
#include "request/collection/CreateSimpleCollectionRequest.h"
#include "request/collection/DescribeCollectionRequest.h"
#include "request/collection/DropCollectionFieldPropertiesRequest.h"
#include "request/collection/DropCollectionPropertiesRequest.h"
#include "request/collection/DropCollectionRequest.h"
#include "request/collection/GetCollectionStatsRequest.h"
#include "request/collection/GetLoadStateRequest.h"
#include "request/collection/HasCollectionRequest.h"
#include "request/collection/ListCollectionsRequest.h"
#include "request/collection/LoadCollectionRequest.h"
#include "request/collection/ReleaseCollectionRequest.h"
#include "request/collection/RenameCollectionRequest.h"
#include "request/database/AlterDatabasePropertiesRequest.h"
#include "request/database/CreateDatabaseRequest.h"
#include "request/database/DescribeDatabaseRequest.h"
#include "request/database/DropDatabasePropertiesRequest.h"
#include "request/database/DropDatabaseRequest.h"
#include "request/database/ListDatabasesRequest.h"
#include "request/dml/DeleteRequest.h"
#include "request/dml/InsertRequest.h"
#include "request/dml/UpsertRequest.h"
#include "request/dql/GetRequest.h"
#include "request/dql/HybridSearchRequest.h"
#include "request/dql/QueryIteratorRequest.h"
#include "request/dql/QueryRequest.h"
#include "request/dql/SearchIteratorRequest.h"
#include "request/dql/SearchRequest.h"
#include "request/index/AlterIndexPropertiesRequest.h"
#include "request/index/CreateIndexRequest.h"
#include "request/index/DescribeIndexRequest.h"
#include "request/index/DropIndexPropertiesRequest.h"
#include "request/index/DropIndexRequest.h"
#include "request/index/ListIndexesRequest.h"
#include "request/partition/CreatePartitionRequest.h"
#include "request/partition/DropPartitionRequest.h"
#include "request/partition/GetPartitionStatsRequest.h"
#include "request/partition/HasPartitionRequest.h"
#include "request/partition/ListPartitionsRequest.h"
#include "request/partition/LoadPartitionsRequest.h"
#include "request/partition/ReleasePartitionsRequest.h"
#include "request/rbac/CreateRoleRequest.h"
#include "request/rbac/CreateUserRequest.h"
#include "request/rbac/DescribeRoleRequest.h"
#include "request/rbac/DropRoleRequest.h"
#include "request/rbac/ListPrivilegeGroupsRequest.h"
#include "request/rbac/ListRolesRequest.h"
#include "request/rbac/ListUsersRequest.h"
#include "request/rbac/PrivilegeGroupRequest.h"
#include "request/rbac/PrivilegeV2Request.h"
#include "request/rbac/PrivilegesOfGroupRequest.h"
#include "request/rbac/RoleUserRequest.h"
#include "request/rbac/UpdatePasswordRequest.h"
#include "request/rbac/UserRequest.h"
#include "request/resourcegroup/CreateResourceGroupRequest.h"
#include "request/resourcegroup/ListResourceGroupsRequest.h"
#include "request/resourcegroup/ResourceGroupRequest.h"
#include "request/resourcegroup/TransferNodeRequest.h"
#include "request/resourcegroup/TransferReplicaRequest.h"
#include "request/resourcegroup/UpdateResourceGroupsRequest.h"
#include "request/utility/CheckHealthRequest.h"
#include "request/utility/CompactRequest.h"
#include "request/utility/FlushRequest.h"
#include "request/utility/GetCompactionRequest.h"
#include "request/utility/ListSegmentsRequest.h"
#include "request/utility/RunAnalyzerRequest.h"
#include "response/alias/DescribeAliasResponse.h"
#include "response/alias/ListAliasesResponse.h"
#include "response/collection/DescribeCollectionResponse.h"
#include "response/collection/GetCollectionStatsResponse.h"
#include "response/collection/GetLoadStateResponse.h"
#include "response/collection/HasCollectionResponse.h"
#include "response/collection/ListCollectionsResponse.h"
#include "response/database/DescribeDatabaseResponse.h"
#include "response/database/ListDatabasesResponse.h"
#include "response/dml/DmlResponse.h"
#include "response/dql/QueryResponse.h"
#include "response/dql/SearchResponse.h"
#include "response/index/DescribeIndexResponse.h"
#include "response/index/ListIndexesResponse.h"
#include "response/partition/GetPartitionStatsResponse.h"
#include "response/partition/HasPartitionResponse.h"
#include "response/partition/ListPartitionsResponse.h"
#include "response/rbac/DescribeRoleResponse.h"
#include "response/rbac/DescribeUserResponse.h"
#include "response/rbac/ListPrivilegeGroupsResponse.h"
#include "response/rbac/ListRolesResponse.h"
#include "response/rbac/ListUsersResponse.h"
#include "response/resourcegroup/DescribeResourceGroupResponse.h"
#include "response/resourcegroup/ListResourceGroupsResponse.h"
#include "response/utility/CheckHealthResponse.h"
#include "response/utility/CompactResponse.h"
#include "response/utility/GetCompactionPlansResponse.h"
#include "response/utility/GetCompactionStateResponse.h"
#include "response/utility/ListSegmentsResponse.h"
#include "response/utility/RunAnalyzerResponse.h"
#include "types/ConnectParam.h"
#include "types/Constants.h"
#include "types/Iterator.h"
#include "types/RetryParam.h"

/**
 *  @brief namespace milvus
 */
namespace milvus {

/**
 * @brief Milvus client abstract class, provide Create() method to create an implementation instance.
 */
class MilvusClientV2 {
 public:
    virtual ~MilvusClientV2() = default;

    /**
     * @brief Crate a MilvusClientV2 instance.
     *
     * @return std::shared_ptr<MilvusClientV2>
     */
    static std::shared_ptr<MilvusClientV2>
    Create();

    /**
     * @brief Connect to Milvus server.
     *
     * @param [in] connect_param server address and port
     * @return Status operation successfully or not
     */
    virtual Status
    Connect(const ConnectParam& connect_param) = 0;

    /**
     * @brief Close connections between client and server.
     *
     * @return Status operation successfully or not
     */
    virtual Status
    Disconnect() = 0;

    /**
     * @brief Change timeout value in milliseconds for each RPC call.
     *
     */
    virtual Status
    SetRpcDeadlineMs(uint64_t timeout_ms) = 0;

    /**
     * @brief Reset retry rules for each RPC call.
     *
     *  @param [in] retry_param retry rules
     */
    virtual Status
    SetRetryParam(const RetryParam& retry_param) = 0;

    /**
     * @brief Get milvus server version.
     *
     * @param [out] version version string
     * @return Status operation successfully or not
     *
     */
    virtual Status
    GetServerVersion(std::string& version) = 0;

    /**
     * @brief Get SDK version.
     *
     * @param [out] version version string
     * @return Status operation successfully or not
     *
     */
    virtual Status
    GetSDKVersion(std::string& version) = 0;

    /**
     * @brief Check healthy of the server.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    CheckHealth(const CheckHealthRequest& request, CheckHealthResponse& response) = 0;

    /**
     * @brief Create a collection.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    CreateCollection(const CreateCollectionRequest& request) = 0;

    /**
     * @brief Create a simple collection with a primary field and a vector field.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    CreateCollection(const CreateSimpleCollectionRequest& request) = 0;

    /**
     * @brief Check existence of a collection.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    HasCollection(const HasCollectionRequest& request, HasCollectionResponse& response) = 0;

    /**
     * @brief Drop a collection, with all its partitions, index and segments.
     *
     * @param [in] request input parameters
     */
    virtual Status
    DropCollection(const DropCollectionRequest& request) = 0;

    /**
     * @brief Load collection data into CPU memory of query node.
     * If the request is sync mode, this api will check collection's loading progress,
     * waiting until the collection completely loaded into query node. Otherwise, it will
     * return immediately.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    LoadCollection(const LoadCollectionRequest& request) = 0;

    /**
     * @brief Release collection data from query node.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    ReleaseCollection(const ReleaseCollectionRequest& request) = 0;

    /**
     * @brief Get collection description, including its schema and properties.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    DescribeCollection(const DescribeCollectionRequest& request, DescribeCollectionResponse& response) = 0;

    /**
     * @brief RenameCollection rename a collection.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    RenameCollection(const RenameCollectionRequest& request) = 0;

    /**
     * @brief Get collection statistics, currently only return row count.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    GetCollectionStats(const GetCollectionStatsRequest& request, GetCollectionStatsResponse& response) = 0;

    /**
     * @brief List all collections brief information.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    ListCollections(const ListCollectionsRequest& request, ListCollectionsResponse& response) = 0;

    /**
     * @brief Get load state of collection or partitions.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    GetLoadState(const GetLoadStateRequest& request, GetLoadStateResponse& response) = 0;

    /**
     * @brief Alter a collection's properties.
     * Read the doc for more info: https://milvus.io/docs/modify-collection.md#Set-Collection-Properties
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    AlterCollectionProperties(const AlterCollectionPropertiesRequest& request) = 0;

    /**
     * @brief Drop a collection's properties.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    DropCollectionProperties(const DropCollectionPropertiesRequest& request) = 0;

    /**
     * @brief Alter a field's properties.
     * Read the doc for more info: https://milvus.io/docs/alter-collection-field.md
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    AlterCollectionFieldProperties(const AlterCollectionFieldPropertiesRequest& request) = 0;

    /**
     * @brief Drop a field's properties.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    DropCollectionFieldProperties(const DropCollectionFieldPropertiesRequest& request) = 0;

    /**
     * @brief Add a field to an existing collection.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    AddCollectionField(const AddCollectionFieldRequest& request) = 0;

    /**
     * @brief Create a partition in a collection.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    CreatePartition(const CreatePartitionRequest& request) = 0;

    /**
     * @brief Drop a partition, with its index and segments.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    DropPartition(const DropPartitionRequest& request) = 0;

    /**
     * @brief Check existence of a partition.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    HasPartition(const HasPartitionRequest& request, HasPartitionResponse& response) = 0;

    /**
     * @brief Load specific partitions data of one collection into query nodes.
     * If the request is sync mode, this api will check partition's loading progress,
     * waiting until all the partitions completely loaded into query node. Otherwise,
     * it will return immediately.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    LoadPartitions(const LoadPartitionsRequest& request) = 0;

    /**
     * @brief Release specific partitions data of one collection into query nodes.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    ReleasePartitions(const ReleasePartitionsRequest& request) = 0;

    /**
     * @brief Get partition statistics, currently only return row count.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    GetPartitionStatistics(const GetPartitionStatsRequest& request, GetPartitionStatsResponse& response) = 0;

    /**
     * @brief List partitions of a collection.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    ListPartitions(const ListPartitionsRequest& request, ListPartitionsResponse& response) = 0;

    /**
     * @brief Create an alias for a collection. Alias can be used in search or query to replace the collection name.
     * For more information: https://wiki.lfaidata.foundation/display/MIL/MEP+10+--+Support+Collection+Alias
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    CreateAlias(const CreateAliasRequest& request) = 0;

    /**
     * @brief Drop an alias.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    DropAlias(const DropAliasRequest& request) = 0;

    /**
     * @brief Change an alias from a collection to another.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    AlterAlias(const AlterAliasRequest& request) = 0;

    /**
     * @brief Describe an alias.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    DescribeAlias(const DescribeAliasRequest& request, DescribeAliasResponse& response) = 0;

    /**
     * @brief List all aliases of a collection.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    ListAliases(const ListAliasesRequest& request, ListAliasesResponse& response) = 0;

    /**
     * @brief Switch connection to another database.
     *
     * @param [in] db_name name of the database
     * @return Status operation successfully or not
     */
    virtual Status
    UseDatabase(const std::string& db_name) = 0;

    /**
     * @brief Get current used database name of the MilvusClient.
     * This API is useful in multi-database scenarios.
     *
     * @param [out] db_name name of the current used database
     * @return Status operation successfully or not
     */
    virtual Status
    CurrentUsedDatabase(std::string& db_name) = 0;

    /**
     * @brief Create a new database.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    CreateDatabase(const CreateDatabaseRequest& request) = 0;

    /**
     * @brief Drop a database.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    DropDatabase(const DropDatabaseRequest& request) = 0;

    /**
     * @brief List all databases.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    ListDatabases(const ListDatabasesRequest& request, ListDatabasesResponse& response) = 0;

    /**
     * @brief Alter a database's properties.
     * Available properties of database are listed here:
     * https://milvus.io/docs/manage_databases.md#Manage-database-properties
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    AlterDatabaseProperties(const AlterDatabasePropertiesRequest& request) = 0;

    /**
     * @brief Drop a database's properties.
     * Available properties of database are listed here:
     * https://milvus.io/docs/manage_databases.md#Manage-database-properties
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    DropDatabaseProperties(const DropDatabasePropertiesRequest& request) = 0;

    /**
     * @brief Describe a database, including its properties.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    DescribeDatabase(const DescribeDatabaseRequest& request, DescribeDatabaseResponse& response) = 0;

    /**
     * @brief Create indexes on vectir fields or scalar fields. You can specify multiple indexes in one call.
     * Read the doc for more info: https://milvus.io/docs/index-explained.md
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    CreateIndex(const CreateIndexRequest& request) = 0;

    /**
     * @brief Get index descriptions and parameters.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    DescribeIndex(const DescribeIndexRequest& request, DescribeIndexResponse& response) = 0;

    /**
     * @brief Get index names of a collection.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    ListIndexes(const ListIndexesRequest& request, ListIndexesResponse& response) = 0;

    /**
     * @brief Drop index of a field.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    DropIndex(const DropIndexRequest& request) = 0;

    /**
     * @brief Alter an index's properties.
     * Read the doc for more info: https://milvus.io/docs/mmap.md#Index-specific-mmap-settings
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    AlterIndexProperties(const AlterIndexPropertiesRequest& request) = 0;

    /**
     * @brief Drop an index's properties.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    DropIndexProperties(const DropIndexPropertiesRequest& request) = 0;

    /**
     * @brief Insert data into a collection.You can input column-based data or row-based data.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    Insert(const InsertRequest& request, InsertResponse& response) = 0;

    /**
     * @brief Upsert entities of a collection.You can input column-based data or row-based data.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    Upsert(const UpsertRequest& request, UpsertResponse& response) = 0;

    /**
     * @brief Delete entities by filtering expression or ID array.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    Delete(const DeleteRequest& request, DeleteResponse& response) = 0;

    /**
     * @brief Search a collection based on the given parameters and return results.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    Search(const SearchRequest& request, SearchResponse& response) = 0;

    /**
     * @brief Get SearchIterator object based on scalar field(s) by filtering expression.
     * Don't disconnect the MilvusClientV2 when the iterator is in using.
     * Note that the order of the returned entities cannot be guaranteed.
     * Read the doc for more info: https://milvus.io/docs/with-iterators.md
     *
     * @param [in] request input parameters, the input is not const because this interface internally need to
     * assign the primary key field name to request.
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    SearchIterator(SearchIteratorRequest& request, SearchIteratorPtr& response) = 0;

    /**
     * @brief Hybrid search a collection based on the given parameters and return results.
     * Read the doc for more info: https://milvus.io/docs/multi-vector-search.md
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    HybridSearch(const HybridSearchRequest& request, HybridSearchResponse& response) = 0;

    /**
     * @brief Query with a set of criteria, and results in a list of records that match the query exactly.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    Query(const QueryRequest& request, QueryResponse& response) = 0;

    /**
     * @brief Query with primary keys, and results in a list of records.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    Get(const GetRequest& request, GetResponse& response) = 0;

    /**
     * @brief Get QueryIterator object based on scalar field(s) by filtering expression.
     * Don't disconnect the MilvusClientV2 when the iterator is in using.
     * Read the doc for more info: https://milvus.io/docs/get-and-scalar-query.md#Use-QueryIterator
     *
     * @param [in] request input parameters, the input is not const because this interface internally need to
     * assign the primary key field name to request.
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    QueryIterator(QueryIteratorRequest& request, QueryIteratorPtr& response) = 0;

    /**
     * @brief Run analyzer. Return result tokens of analysis.
     * Milvus server supports this interface from v2.5.11.
     * Read the doc for more info: https://milvus.io/docs/analyzer-overview.md
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    RunAnalyzer(const RunAnalyzerRequest& request, RunAnalyzerResponse& response) = 0;

    /**
     * @brief Flush insert buffer data into storage.
     * If the FlushRequest.WaitFlushedMs is larger than zero, it will check related segments state in a loop,
     * to make sure the data persisted successfully.
     * Flush is a heavy operation, it is not recommended to call it frequently. Just let the milvus server
     * automatically triggers flush action.
     * By default, the call frequency of Flush() is limited by milvus server-side rate-limit configuration.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    Flush(const FlushRequest& request) = 0;

    /**
     * @brief Retrieve information of persisted segments from data nodes.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    ListPersistentSegments(const ListPersistentSegmentsRequest& request, ListPersistentSegmentsResponse& response) = 0;

    /**
     * @brief Retrieve information of loaded segments from query nodes.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    ListQuerySegments(const ListQuerySegmentsRequest& request, ListQuerySegmentsResponse& response) = 0;

    /**
     * @brief Manually trigger a compaction action.
     * Normally, user no need to call this API sicne milvus automatically triggers compactions internally.
     * It is mainly used for some maintainance or debug purpose.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    Compact(const CompactRequest& request, CompactResponse& response) = 0;

    /**
     * @brief Get compaction action state.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    GetCompactionState(const GetCompactionStateRequest& request, GetCompactionStateResponse& response) = 0;

    /**
     * @brief Get plans of a compaction action.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    GetCompactionPlans(const GetCompactionPlansRequest& request, GetCompactionPlansResponse& response) = 0;

    /**
     * @brief Create a resource group. A resource group to physically isolate certain query nodes from others.
     * Read the doc for more info: https://milvus.io/docs/resource_group.md#Manage-Resource-Groups
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    CreateResourceGroup(const CreateResourceGroupRequest& request) = 0;

    /**
     * @brief Drop a resource group.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    DropResourceGroup(const DropResourceGroupRequest& request) = 0;

    /**
     * @brief Update resource groups.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    UpdateResourceGroups(const UpdateResourceGroupsRequest& request) = 0;

    /**
     * @brief Transfer query nodes to another resource groups.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    TransferNode(const TransferNodeRequest& request) = 0;

    /**
     * @brief Transfer replicas of a collection from a resource group to another.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    TransferReplica(const TransferReplicaRequest& request) = 0;

    /**
     * @brief List all the resource groups under the current database.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    ListResourceGroups(const ListResourceGroupsRequest& request, ListResourceGroupsResponse& response) = 0;

    /**
     * @brief Describe a resource group.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    DescribeResourceGroup(const DescribeResourceGroupRequest& request, DescribeResourceGroupResponse& response) = 0;

    /**
     * @brief Create an user with username and password to login milvus.
     * Read the doc for more info: https://milvus.io/docs/users_and_roles.md
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    CreateUser(const CreateUserRequest& request) = 0;

    /**
     * @brief Update password of an user.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    UpdatePassword(const UpdatePasswordRequest& request) = 0;

    /**
     * @brief Drop an user.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    DropUser(const DropUserRequest& request) = 0;

    /**
     * @brief Describe an user.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    DescribeUser(const DescribeUserRequest& request, DescribeUserResponse& response) = 0;

    /**
     * @brief List users.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    ListUsers(const ListUsersRequest& request, ListUsersResponse& response) = 0;

    /**
     * @brief Create a role with specific privileges.
     * Read the doc for more info: https://milvus.io/docs/users_and_roles.md
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    CreateRole(const CreateRoleRequest& request) = 0;

    /**
     * @brief Drop a role.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    DropRole(const DropRoleRequest& request) = 0;

    /**
     * @brief Describe an role.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    DescribeRole(const DescribeRoleRequest& request, DescribeRoleResponse& response) = 0;

    /**
     * @brief List roles.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    ListRoles(const ListRolesRequest& request, ListRolesResponse& response) = 0;

    /**
     * @brief Grant a role to an user.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    GrantRole(const GrantRoleRequest& request) = 0;

    /**
     * @brief Revoke a role from an user.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    RevokeRole(const RevokeRoleRequest& request) = 0;

    /**
     * @brief Grant a privilege or a privilege group to a role.
     * This is V2 proto interface, the V1 interface is no longer used.
     * Read the doc for more info: https://milvus.io/docs/v2.5.x/grant_privileges.md
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    GrantPrivilegeV2(const GrantPrivilegeV2Request& request) = 0;

    /**
     * @brief Revoke a privilege or a privilege group from a role.
     * This is V2 proto interface, the V1 interface is no longer used.
     * Read the doc for more info: https://milvus.io/docs/v2.5.x/grant_privileges.md
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    RevokePrivilegeV2(const RevokePrivilegeV2Request& request) = 0;

    /**
     * @brief Create a privilege group.
     * Read the doc for more info: https://milvus.io/docs/privilege_group.md
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    CreatePrivilegeGroup(const CreatePrivilegeGroupRequest& request) = 0;

    /**
     * @brief Drop a privilege group.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    DropPrivilegeGroup(const DropPrivilegeGroupRequest& request) = 0;

    /**
     * @brief List all the privilege groups.
     *
     * @param [in] request input parameters
     * @param [out] response output results
     * @return Status operation successfully or not
     */
    virtual Status
    ListPrivilegeGroups(const ListPrivilegeGroupsRequest& request, ListPrivilegeGroupsResponse& response) = 0;

    /**
     * @brief Add privileges to a privilege group.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    AddPrivilegesToGroup(const AddPrivilegesToGroupRequest& request) = 0;

    /**
     * @brief Remove privileges from a privilege group.
     *
     * @param [in] request input parameters
     * @return Status operation successfully or not
     */
    virtual Status
    RemovePrivilegesFromGroup(const RemovePrivilegesFromGroupRequest& request) = 0;
};

using MilvusClientV2Ptr = std::shared_ptr<MilvusClientV2>;

}  // namespace milvus
