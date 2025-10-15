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

#include <memory>

#include "Status.h"
#include "types/AliasDesc.h"
#include "types/CollectionDesc.h"
#include "types/CollectionInfo.h"
#include "types/CollectionSchema.h"
#include "types/CollectionStat.h"
#include "types/CompactionPlan.h"
#include "types/CompactionState.h"
#include "types/ConnectParam.h"
#include "types/Constants.h"
#include "types/DatabaseDesc.h"
#include "types/DmlResults.h"
#include "types/FieldData.h"
#include "types/HybridSearchArguments.h"
#include "types/IndexDesc.h"
#include "types/IndexState.h"
#include "types/Iterator.h"
#include "types/IteratorArguments.h"
#include "types/PartitionInfo.h"
#include "types/PartitionStat.h"
#include "types/PrivilegeGroupInfo.h"
#include "types/ProgressMonitor.h"
#include "types/QueryArguments.h"
#include "types/QueryResults.h"
#include "types/ResourceGroupConfig.h"
#include "types/ResourceGroupDesc.h"
#include "types/RetryParam.h"
#include "types/RoleDesc.h"
#include "types/SearchArguments.h"
#include "types/SearchResults.h"
#include "types/SegmentInfo.h"
#include "types/UserDesc.h"

/**
 *  @brief namespace milvus
 */
namespace milvus {

/**
 * @brief Milvus client abstract class, provide Create() method to create an implementation instance.
 */
class MilvusClient {
 public:
    virtual ~MilvusClient() = default;

    /**
     * @brief Crate a MilvusClient instance.
     *
     * @return std::shared_ptr<MilvusClient>
     */
    static std::shared_ptr<MilvusClient>
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
     * @brief Break connections between client and server.
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
     * @deprecated replaced by GetServerVersion()
     *
     * @param [out] version version string
     * @return Status operation successfully or not
     *
     */
    virtual Status
    GetVersion(std::string& version) = 0;

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
     * @brief Create a collection with schema.
     *
     * @param [in] schema schema of the collection
     * @param [in] num_partitions num of default physical partitions, only used in partition key mode
     *  and changes are not supported, default value is 16 if set to negative or zero.
     * @return Status operation successfully or not
     */
    virtual Status
    CreateCollection(const CollectionSchema& schema, int64_t num_partitions = 0) = 0;

    /**
     * @brief Check existence of a collection.
     *
     * @param [in] collection_name name of the collection
     * @param [out] has true: collection exists, false: collection doesn't exist
     * @return Status operation successfully or not
     */
    virtual Status
    HasCollection(const std::string& collection_name, bool& has) = 0;

    /**
     * @brief Drop a collection, with all its partitions, index and segments.
     *
     * @param [in] collection_name name of the collection
     * @return Status operation successfully or not
     */
    virtual Status
    DropCollection(const std::string& collection_name) = 0;

    /**
     * @brief Load collection data into CPU memory of query node.
     * This api will check collection's loading progress,
     * waiting until the collection completely loaded into query node.
     *
     * @param [in] collection_name name of the collection
     * @param [in] replica_number the number of replicas, default 1
     * @param [in] progress_monitor set timeout to wait loading progress complete, set to ProgressMonitor::NoWait() to
     * return instantly, set to ProgressMonitor::Forever() to wait until finished.
     * @return Status operation successfully or not
     */
    virtual Status
    LoadCollection(const std::string& collection_name, int replica_number = 1,
                   const ProgressMonitor& progress_monitor = ProgressMonitor::Forever()) = 0;

    /**
     * @brief Release collection data from query node.
     *
     * @param [in] collection_name name of the collection
     * @return Status operation successfully or not
     */
    virtual Status
    ReleaseCollection(const std::string& collection_name) = 0;

    /**
     * @brief Get collection description, including its schema.
     *
     * @param [in] collection_name name of the collection
     * @param [out] collection_desc collection's description
     * @return Status operation successfully or not
     */
    virtual Status
    DescribeCollection(const std::string& collection_name, CollectionDesc& collection_desc) = 0;

    /**
     * @brief RenameCollection rename a collection.
     *
     * @param [in] collection_name name of the collection
     * @param [in] new_collection_name new name of the collection
     * @return Status operation successfully or not
     */
    virtual Status
    RenameCollection(const std::string& collection_name, const std::string& new_collection_name) = 0;

    /**
     * @brief Get collection statistics, currently only return row count.
     * If the timeout is specified, this api will call Flush() and wait all segments persisted into storage.
     *
     * @param [in] collection_name name of the collection
     * @param [in] progress_monitor set timeout to wait flush progress complete, set to ProgressMonitor::NoWait() to
     * return instantly, set to ProgressMonitor::Forever() to wait until finished.
     * @param [out] collection_stat statistics of the collection
     * @return Status operation successfully or not
     */
    virtual Status
    GetCollectionStatistics(const std::string& collection_name, CollectionStat& collection_stat,
                            const ProgressMonitor& progress_monitor = ProgressMonitor::Forever()) = 0;

    /**
     * @brief If the collection_names is empty, list all collections brief information's.
     * If the collection_names is specified, return the specified collection's loading process state.
     * @deprecated In v2.4, the parameter collection_names is no longer work, use the ListCollections() instead.
     *
     * @param [in] collection_names name array of collections
     * @param [out] collections_info brief information's of the collections
     * @return Status operation successfully or not
     */
    virtual Status
    ShowCollections(const std::vector<std::string>& collection_names, CollectionsInfo& collections_info) = 0;

    /**
     * @brief List all collections brief information's.
     *
     * @param [out] collections_info brief information's of the collections
     * @param [in] only_show_loaded set to true only show in-memory collections, otherwise show all collections.
     * @return Status operation successfully or not
     */
    virtual Status
    ListCollections(CollectionsInfo& collections_info, bool only_show_loaded = false) = 0;

    /**
     * @brief Get load state of collection or partitions.
     *
     * @param [in] collection_name name of the collection
     * @param [in] partition_names name array of the partitions
     * @param [out] is_loaded whether the collection is loaded into memory
     * @return Status operation successfully or not
     */
    virtual Status
    GetLoadState(const std::string& collection_name, bool& is_loaded,
                 const std::vector<std::string> partition_names = {}) = 0;

    /**
     * @brief Alter a collection's properties.
     *
     * @param [in] collection_name name of the collection
     * @param [in] properties properties to be altered
     * @return Status operation successfully or not
     */
    virtual Status
    AlterCollectionProperties(const std::string& collection_name,
                              const std::unordered_map<std::string, std::string>& properties) = 0;

    /**
     * @brief Drop a collection's properties.
     *
     * @param [in] collection_name name of the collection
     * @param [in] property_keys keys of the properties
     * @return Status operation successfully or not
     */
    virtual Status
    DropCollectionProperties(const std::string& collection_name, const std::set<std::string>& property_keys) = 0;

    /**
     * @brief Alter a field's properties.
     *
     * @param [in] collection_name name of the collection
     * @param [in] field_name name of the field
     * @param [in] properties properties to be altered
     * @return Status operation successfully or not
     */
    virtual Status
    AlterCollectionField(const std::string& collection_name, const std::string& field_name,
                         const std::unordered_map<std::string, std::string>& properties) = 0;

    /**
     * @brief Create a partition in a collection.
     *
     * @param [in] collection_name name of the collection
     * @param [in] partition_name name of the partition
     * @return Status operation successfully or not
     */
    virtual Status
    CreatePartition(const std::string& collection_name, const std::string& partition_name) = 0;

    /**
     * @brief Drop a partition, with its index and segments.
     *
     * @param [in] collection_name name of the collection
     * @param [in] partition_name name of the partition
     * @return Status operation successfully or not
     */
    virtual Status
    DropPartition(const std::string& collection_name, const std::string& partition_name) = 0;

    /**
     * @brief Check existence of a partition.
     *
     * @param [in] collection_name name of the collection
     * @param [in] partition_name name of the partition
     * @param [out] has true: partition exists, false: partition doesn't exist
     * @return Status operation successfully or not
     */
    virtual Status
    HasPartition(const std::string& collection_name, const std::string& partition_name, bool& has) = 0;

    /**
     * @brief Load specific partitions data of one collection into query nodes.
     * This api will check partition's loading progress,
     * waiting until all the partitions completely loaded into query node.
     *
     * @param [in] collection_name name of the collection
     * @param [in] partition_names name array of the partitions
     * @param [in] replica_number the number of replicas, default 1
     * @param [in] progress_monitor set timeout to wait loading progress complete, set to
     * ProgressMonitor::NoWait() to return instantly, set to ProgressMonitor::Forever() to wait until finished.
     * @return Status operation successfully or not
     */
    virtual Status
    LoadPartitions(const std::string& collection_name, const std::vector<std::string>& partition_names,
                   int replica_number = 1, const ProgressMonitor& progress_monitor = ProgressMonitor::Forever()) = 0;

    /**
     * @brief Release specific partitions data of one collection into query nodes.
     *
     * @param [in] collection_name name of the collection
     * @param [in] partition_names name array of the partitions
     * @return Status operation successfully or not
     */
    virtual Status
    ReleasePartitions(const std::string& collection_name, const std::vector<std::string>& partition_names) = 0;

    /**
     * @brief Get partition statistics, currently only return row count.
     * If the timeout is specified, this api will call Flush() and wait all segments persisted into storage.
     *
     * @param [in] collection_name name of the collection
     * @param [in] partition_name name of the partition
     * @param [in] progress_monitor set timeout to wait flush progress complete, set to ProgressMonitor::NoWait() to
     * return instantly, set to ProgressMonitor::Forever() to wait until finished.
     * @param [out] partition_stat statistics of the partition
     * @return Status operation successfully or not
     */
    virtual Status
    GetPartitionStatistics(const std::string& collection_name, const std::string& partition_name,
                           PartitionStat& partition_stat,
                           const ProgressMonitor& progress_monitor = ProgressMonitor::Forever()) = 0;

    /**
     * @brief If the partition_names is empty, list all partitions brief information's.
     * If the partition_names is specified, return the specified partition's loading process state.
     * @deprecated In v2.4, the parameter partition_names is no longer work, use the ListPartitions() instead.
     *
     * @param [in] collection_name name of the collection
     * @param [in] partition_names name array of the partitions
     * @param [out] partitions_info brief information's of the partitions
     * @return Status operation successfully or not
     */
    virtual Status
    ShowPartitions(const std::string& collection_name, const std::vector<std::string>& partition_names,
                   PartitionsInfo& partitions_info) = 0;

    /**
     * @brief If the partition_names is empty, list all partitions brief information's.
     *
     * @param [in] collection_name name of the collection
     * @param [out] partitions_info brief information's of the partitions
     * @param [in] only_show_loaded set to ture only show in-memory partitions, otherwise show all partitions.
     * @return Status operation successfully or not
     */
    virtual Status
    ListPartitions(const std::string& collection_name, PartitionsInfo& partitions_info,
                   bool only_show_loaded = false) = 0;

    /**
     * @brief Create an alias for a collection. Alias can be used in search or query to replace the collection name.
     * For more information: https://wiki.lfaidata.foundation/display/MIL/MEP+10+--+Support+Collection+Alias
     *
     * @param [in] collection_name name of the collection
     * @param [in] alias alias of the partitions
     * @return Status operation successfully or not
     */
    virtual Status
    CreateAlias(const std::string& collection_name, const std::string& alias) = 0;

    /**
     * @brief Drop an alias.
     *
     * @param [in] alias alias of the partitions
     * @return Status operation successfully or not
     */
    virtual Status
    DropAlias(const std::string& alias) = 0;

    /**
     * @brief Change an alias from a collection to another.
     *
     * @param [in] collection_name name of the collection
     * @param [in] alias alias of the partitions
     * @return Status operation successfully or not
     */
    virtual Status
    AlterAlias(const std::string& collection_name, const std::string& alias) = 0;

    /**
     * @brief Describe an alias.
     *
     * @param [in] alias_name name of the alias
     * @return Status operation successfully or not
     */
    virtual Status
    DescribeAlias(const std::string& alias_name, AliasDesc& desc) = 0;

    /**
     * @brief List all aliases of a collection.
     *
     * @param [in] collection_name name of the collection
     * @param [out] descs a list of aliases
     * @return Status operation successfully or not
     */
    virtual Status
    ListAliases(const std::string& collection_name, std::vector<AliasDesc>& descs) = 0;

    /**
     * @brief Switch connection to another database.
     *
     * @param [in] db_name name of the database
     * @return Status operation successfully or not
     */
    virtual Status
    UseDatabase(const std::string& db_name) = 0;

    /**
     * @brief Get current used database name.
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
     * @param [in] db_name name of the database
     * @param [in] properties properties of the database, available keys of properties
     *   are listed here: https://milvus.io/docs/manage_databases.md#Manage-database-properties
     * @return Status operation successfully or not
     */
    virtual Status
    CreateDatabase(const std::string& db_name, const std::unordered_map<std::string, std::string>& properties) = 0;

    /**
     * @brief Drop a database.
     *
     * @param [in] db_name name of the database
     * @return Status operation successfully or not
     */
    virtual Status
    DropDatabase(const std::string& db_name) = 0;

    /**
     * @brief Drop a database.
     *
     * @param [out] names a list of database names
     * @return Status operation successfully or not
     */
    virtual Status
    ListDatabases(std::vector<std::string>& names) = 0;

    /**
     * @brief Alter a database's properties.
     *
     * @param [in] db_name name of the database
     * @param [in] properties properties of the database to be updated, available keys of properties
     *   are listed here: https://milvus.io/docs/manage_databases.md#Manage-database-properties
     * @return Status operation successfully or not
     */
    virtual Status
    AlterDatabaseProperties(const std::string& db_name,
                            const std::unordered_map<std::string, std::string>& properties) = 0;

    /**
     * @brief Drop a database's properties.
     *
     * @param [in] db_name name of the database
     * @param [in] properties properties of the database to be deleted, available keys of properties
     *   are listed here: https://milvus.io/docs/manage_databases.md#Manage-database-properties
     * @return Status operation successfully or not
     */
    virtual Status
    DropDatabaseProperties(const std::string& db_name, const std::vector<std::string>& properties) = 0;

    /**
     * @brief Describe a database.
     *
     * @param [in] db_name name of the database
     * @param [out] db_desc information of the database
     * @return Status operation successfully or not
     */
    virtual Status
    DescribeDatabase(const std::string& db_name, DatabaseDesc& db_desc) = 0;

    /**
     * @brief Create an index on a field. Currently only support index on vector field.
     *
     * @param [in] collection_name name of the collection
     * @param [in] index_desc the index descriptions and parameters
     * @param [in] progress_monitor set timeout to wait index progress complete, set to ProgressMonitor::NoWait() to
     * return instantly, set to ProgressMonitor::Forever() to wait until finished.
     * @return Status operation successfully or not
     */
    virtual Status
    CreateIndex(const std::string& collection_name, const IndexDesc& index_desc,
                const ProgressMonitor& progress_monitor = ProgressMonitor::Forever()) = 0;

    /**
     * @brief Get index descriptions and parameters.
     *
     * @param [in] collection_name name of the collection
     * @param [in] field_name name of the field
     * @param [out] index_desc index descriptions and parameters
     * @return Status operation successfully or not
     */
    virtual Status
    DescribeIndex(const std::string& collection_name, const std::string& field_name, IndexDesc& index_desc) = 0;

    /**
     * @brief Get index names of a collection.
     *
     * @param [in] collection_name name of the collection
     * @param [in] field_name name of the field, if this value is empty, return all index names
     * @param [out] index_names index names
     * @return Status operation successfully or not
     */
    virtual Status
    ListIndexes(const std::string& collection_name, const std::string& field_name,
                std::vector<std::string>& index_names) = 0;

    /**
     * @brief Get state of an index. From the state client can know whether the index has finished or in-progress.
     *
     * @param [in] collection_name name of the collection
     * @param [in] field_name name of the field
     * @param [out] state index state of field
     * @return Status operation successfully or not
     */
    virtual Status
    GetIndexState(const std::string& collection_name, const std::string& field_name, IndexState& state) = 0;

    /**
     * @brief Get progress of an index. From the progress client can how many rows have been indexed.
     *
     * @param [in] collection_name name of the collection
     * @param [in] field_name name of the field
     * @param [out] progress progress array of field, currently only return one index progress
     * @return Status operation successfully or not
     */
    virtual Status
    GetIndexBuildProgress(const std::string& collection_name, const std::string& field_name,
                          IndexProgress& progress) = 0;

    /**
     * @brief Drop index of a field.
     *
     * @param [in] collection_name name of the collection
     * @param [in] field_name name of the field
     * @return Status operation successfully or not
     */
    virtual Status
    DropIndex(const std::string& collection_name, const std::string& field_name) = 0;

    /**
     * @brief Alter an index's properties.
     *
     * @param [in] collection_name name of the collection
     * @param [in] index_name name of the index
     * @param [in] properties properties to be altered
     * @return Status operation successfully or not
     */
    virtual Status
    AlterIndexProperties(const std::string& collection_name, const std::string& index_name,
                         const std::unordered_map<std::string, std::string>& properties) = 0;

    /**
     * @brief Drop an index's properties.
     *
     * @param [in] collection_name name of the collection
     * @param [in] index_name name of the index
     * @param [in] property_keys keys of the properties
     * @return Status operation successfully or not
     */
    virtual Status
    DropIndexProperties(const std::string& collection_name, const std::string& index_name,
                        const std::set<std::string>& property_keys) = 0;

    /**
     * @brief Insert data into a collection by column-based.
     *
     * @param [in] collection_name name of the collection
     * @param [in] partition_name name of the partition, optional(pass an empty string to skip)
     * @param [in] fields insert data
     * @param [out] results insert results
     * @return Status operation successfully or not
     */
    virtual Status
    Insert(const std::string& collection_name, const std::string& partition_name,
           const std::vector<FieldDataPtr>& fields, DmlResults& results) = 0;

    /**
     * @brief Insert data into a collection by row-based.
     *
     * @param [in] collection_name name of the collection
     * @param [in] partition_name name of the partition, optional(pass an empty string to skip)
     * @param [in] rows insert rows
     * @param [out] results insert results
     * @return Status operation successfully or not
     */
    virtual Status
    Insert(const std::string& collection_name, const std::string& partition_name, const EntityRows& rows,
           DmlResults& results) = 0;

    /**
     * @brief Upsert entities into a collection.
     *
     * @param [in] collection_name name of the collection
     * @param [in] partition_name name of the partition, optional(pass an empty string to skip)
     * @param [in] fields upsedrt data
     * @param [out] results upsedrt results
     * @return Status operation successfully or not
     */
    virtual Status
    Upsert(const std::string& collection_name, const std::string& partition_name,
           const std::vector<FieldDataPtr>& fields, DmlResults& results) = 0;

    /**
     * @brief Upsert rows into a collection.
     *
     * @param [in] collection_name name of the collection
     * @param [in] partition_name name of the partition, optional(pass an empty string to skip)
     * @param [in] fields upsedrt rows
     * @param [out] results upsedrt results
     * @return Status operation successfully or not
     */
    virtual Status
    Upsert(const std::string& collection_name, const std::string& partition_name, const EntityRows& rows,
           DmlResults& results) = 0;

    /**
     * @brief Delete entities by filtering condition.
     *
     * @param [in] collection_name name of the collection
     * @param [in] partition_name name of the partition, optional(pass an empty string to skip)
     * @param [in] expression the expression to filter out entities, currently only support primary key as filtering.
     * For example: "id in [1, 2, 3]"
     * @param [out] results insert results
     * @return Status operation successfully or not
     */
    virtual Status
    Delete(const std::string& collection_name, const std::string& partition_name, const std::string& expression,
           DmlResults& results) = 0;

    /**
     * @brief Search a collection based on the given parameters and return results.
     *
     * @param [in] arguments search arguments
     * @param [out] results search results
     * @return Status operation successfully or not
     */
    virtual Status
    Search(const SearchArguments& arguments, SearchResults& results) = 0;

    /**
     * @brief Get SearchIterator object based on scalar field(s) filtered by boolean expression.
     * Note that the order of the returned entities cannot be guaranteed.
     *
     * @param [in] arguments search iterator arguments
     * @param [out] iterator search iterator object
     * @return Status operation successfully or not
     */
    virtual Status
    SearchIterator(SearchIteratorArguments& arguments, SearchIteratorPtr& iterator) = 0;

    /**
     * @brief Hybrid search a collection based on the given parameters and return results.
     *
     * @param [in] arguments search arguments
     * @param [out] results search results
     * @return Status operation successfully or not
     */
    virtual Status
    HybridSearch(const HybridSearchArguments& arguments, SearchResults& results) = 0;

    /**
     * @brief Query with a set of criteria, and results in a list of records that match the query exactly.
     *
     * @param [in] arguments query arguments
     * @param [out] results query results
     * @return Status operation successfully or not
     */
    virtual Status
    Query(const QueryArguments& arguments, QueryResults& results) = 0;

    /**
     * @brief Get QueryIterator object based on scalar field(s) filtered by boolean expression.
     *
     * @param [in] arguments query iterator arguments
     * @param [out] iterator query iterator object
     * @return Status operation successfully or not
     */
    virtual Status
    QueryIterator(QueryIteratorArguments& arguments, QueryIteratorPtr& iterator) = 0;

    /**
     * @brief Flush insert buffer into storage.
     * To make sure the buffer persisted successfully, it calls GetFlushState() to check related segments state.
     *
     * @param [in] collection_names specify target collection names, if this array is empty, will flush all collections
     * @param [in] progress_monitor timeout setting for waiting progress. Set ProgressMonitor::NoWait() to return
     * instantly, set to ProgressMonitor::Forever() to wait until finished.
     * @return Status operation successfully or not
     */
    virtual Status
    Flush(const std::vector<std::string>& collection_names,
          const ProgressMonitor& progress_monitor = ProgressMonitor::Forever()) = 0;

    /**
     * @brief Get flush state of specified segments.
     *
     * @param [in] segments id array of segments
     * @param [out] flushed true: all the segments has been flushed, false: still in flush progress
     * @return Status operation successfully or not
     */
    virtual Status
    GetFlushState(const std::vector<int64_t>& segments, bool& flushed) = 0;

    /**
     * @brief Retrieve information of persistent segments from data nodes.
     *
     * @param [in] collection_name name of the collection
     * @param [out] segments_info information array for persistent segments
     * @return Status operation successfully or not
     */
    virtual Status
    GetPersistentSegmentInfo(const std::string& collection_name, SegmentsInfo& segments_info) = 0;

    /**
     * @brief Retrieve information of segments from query nodes.
     *
     * @param [in] collection_name name of the collection
     * @param [out] segments_info information array for segments
     * @return Status operation successfully or not
     */
    virtual Status
    GetQuerySegmentInfo(const std::string& collection_name, QuerySegmentsInfo& segments_info) = 0;

    /**
     * @brief Get server runtime statistics.
     *
     * @param [in] request request in json format
     * @param [out] response response in json format
     * @param [out] component_name metrics from which component
     * @return Status operation successfully or not
     */
    virtual Status
    GetMetrics(const std::string& request, std::string& response, std::string& component_name) = 0;

    /**
     * @brief Rebalanced sealed segments from one query node to others.
     *
     * @param [in] src_node the source query node id
     * @param [in] dst_nodes the destiny query nodes id array
     * @param [in] segments the segments id array to be balanced
     * @return Status operation successfully or not
     */
    virtual Status
    LoadBalance(int64_t src_node, const std::vector<int64_t>& dst_nodes, const std::vector<int64_t>& segments) = 0;

    /**
     * @brief Get compaction action state.
     *
     * @param [in] compaction_id the compaction action id
     * @param [out] compaction_state state of the compaction action
     * @return Status operation successfully or not
     */
    virtual Status
    GetCompactionState(int64_t compaction_id, CompactionState& compaction_state) = 0;

    /**
     * @brief Manually trigger a compaction action.
     *
     * @param [in] collection_name name of the collection
     * @param [in] travel_timestamp specify a timestamp to compact on a data view at a specified point in time.
     * @param [out] compaction_id id of the compaction action
     * @return Status operation successfully or not
     */
    virtual Status
    ManualCompaction(const std::string& collection_name, uint64_t travel_timestamp, int64_t& compaction_id) = 0;

    /**
     * @brief Get plans of a compaction action.
     *
     * @param [in] compaction_id the compaction action id
     * @param [out] plans compaction plan array
     * @return Status operation successfully or not
     */
    virtual Status
    GetCompactionPlans(int64_t compaction_id, CompactionPlans& plans) = 0;

    /**
     * @brief Create Credential.
     * @deprecated replaced by CreateUser() in v2.4
     *
     * @param [in] username the username for created
     * @param [in] password the password for the user to be created
     * @return Status operation successfully or not
     */
    virtual Status
    CreateCredential(const std::string& username, const std::string& password) = 0;

    /**
     * @brief Update Credential.
     * @deprecated replaced by UpdatePassword() in v2.4
     *
     * @param [in] username the username for updated
     * @param [in] old_password the old password for the user
     * @param [in] new_password the updated password for the user
     * @return Status operation successfully or not
     */
    virtual Status
    UpdateCredential(const std::string& username, const std::string& old_password, const std::string& new_password) = 0;

    /**
     * @brief Delete Credential.
     * @deprecated replaced by DropUser() in v2.4
     *
     * @param [in] username the username to be deleted
     * @return Status operation successfully or not
     */
    virtual Status
    DeleteCredential(const std::string& username) = 0;

    /**
     * @brief List Users.
     * @deprecated replaced by ListUsers() in v2.4
     *
     * @param [out] the usernames
     * @return Status operation successfully or not
     */
    virtual Status
    ListCredUsers(std::vector<std::string>& names) = 0;

    /**
     * @brief Create a resource group.
     *
     * @param [in] name name of the resource group
     * @param [in] config configurations of the resource group
     * @return Status operation successfully or not
     */
    virtual Status
    CreateResourceGroup(const std::string& name, const ResourceGroupConfig& config) = 0;

    /**
     * @brief Drop a resource group.
     *
     * @param [in] name name of the resource group
     * @return Status operation successfully or not
     */
    virtual Status
    DropResourceGroup(const std::string& name) = 0;

    /**
     * @brief Update resource groups.
     *
     * @param [in] name name of the resource group
     * @param [in] groups configurations of the resource groups
     * @return Status operation successfully or not
     */
    virtual Status
    UpdateResourceGroups(const std::unordered_map<std::string, ResourceGroupConfig>& groups) = 0;

    /**
     * @brief Transfer nodes to another resource groups.
     *
     * @param [in] source_group name of the source resource group
     * @param [in] target_group name of the target resource group
     * @param [in] num_nodes number of nodes to be transfered
     * @return Status operation successfully or not
     */
    virtual Status
    TransferNode(const std::string& source_group, const std::string& target_group, uint32_t num_nodes) = 0;

    /**
     * @brief Transfer replicas of a collection from source group to target group.
     *
     * @param [in] source_group name of the source resource group
     * @param [in] target_group name of the target resource group
     * @param [in] collection_name name of a collection
     * @param [in] num_replicas number of replicas to be transfered
     * @return Status operation successfully or not
     */
    virtual Status
    TransferReplica(const std::string& source_group, const std::string& target_group,
                    const std::string& collection_name, uint32_t num_replicas) = 0;

    /**
     * @brief List all the resource groups under the current database.
     *
     * @param [out] group_names names of the resource groups
     * @return Status operation successfully or not
     */
    virtual Status
    ListResourceGroups(std::vector<std::string>& group_names) = 0;

    /**
     * @brief Describe a resource group.
     *
     * @param [in] group_name name of the resource group
     * @param [out] desc details of the resource group
     * @return Status operation successfully or not
     */
    virtual Status
    DescribeResourceGroup(const std::string& group_name, ResourceGroupDesc& desc) = 0;

    /**
     * @brief Create an user.
     *
     * @param [in] user_name name of the user
     * @param [in] password password of the user
     * @return Status operation successfully or not
     */
    virtual Status
    CreateUser(const std::string& user_name, const std::string& password) = 0;

    /**
     * @brief Update password of an user.
     *
     * @param [in] user_name name of the user
     * @param [in] old_password the old password for the user
     * @param [in] new_password the updated password for the user
     * @return Status operation successfully or not
     */
    virtual Status
    UpdatePassword(const std::string& user_name, const std::string& old_password, const std::string& new_password) = 0;

    /**
     * @brief Drop an user.
     *
     * @param [in] user_name name of the user
     * @return Status operation successfully or not
     */
    virtual Status
    DropUser(const std::string& user_name) = 0;

    /**
     * @brief Describe an user.
     *
     * @param [in] user_name name of the user
     * @param [out] desc description of the user
     * @return Status operation successfully or not
     */
    virtual Status
    DescribeUser(const std::string& user_name, UserDesc& desc) = 0;

    /**
     * @brief List users.
     *
     * @param [out] names names of users
     * @return Status operation successfully or not
     */
    virtual Status
    ListUsers(std::vector<std::string>& names) = 0;

    /**
     * @brief Create a role.
     *
     * @param [in] role_name name of the role
     * @return Status operation successfully or not
     */
    virtual Status
    CreateRole(const std::string& role_name) = 0;

    /**
     * @brief Drop a role.
     *
     * @param [in] role_name name of the role
     * @param [in] force_drop force to drop the role even if there is permission binding
     * @return Status operation successfully or not
     */
    virtual Status
    DropRole(const std::string& role_name, bool force_drop = false) = 0;

    /**
     * @brief Describe an role.
     *
     * @param [in] role_name name of the role
     * @param [out] desc description of the role
     * @return Status operation successfully or not
     */
    virtual Status
    DescribeRole(const std::string& role_name, RoleDesc& desc) = 0;

    /**
     * @brief List roles.
     *
     * @param [out] names names of roles
     * @return Status operation successfully or not
     */
    virtual Status
    ListRoles(std::vector<std::string>& names) = 0;

    /**
     * @brief Grant a role to an user.
     *
     * @param [in] user_name name of the user
     * @param [in] role_name name of the role
     * @return Status operation successfully or not
     */
    virtual Status
    GrantRole(const std::string& user_name, const std::string& role_name) = 0;

    /**
     * @brief Revoke a role from an user.
     *
     * @param [in] user_name name of the user
     * @param [in] role_name name of the role
     * @return Status operation successfully or not
     */
    virtual Status
    RevokeRole(const std::string& user_name, const std::string& role_name) = 0;

    /**
     * @brief Grant a privilege or a privilege group to a role.
     * For more info: https://milvus.io/docs/v2.4.x/users_and_roles.md
     *
     * @param [in] role_name name of the role
     * @param [in] privilege the privilege or privilege group to grant
     * @param [in] collection_name name of a collection, "*" for all collections
     * @param [in] db_name name of a database, "*" for all databases
     * @return Status operation successfully or not
     */
    virtual Status
    GrantPrivilege(const std::string& role_name, const std::string& privilege, const std::string& collection_name,
                   const std::string& db_name) = 0;

    /**
     * @brief Revoke a privilege or a privilege group from a role.
     * For more info: https://milvus.io/docs/v2.4.x/users_and_roles.md
     *
     * @param [in] role_name name of the role
     * @param [in] privilege the privilege or privilege group to revoke
     * @param [in] collection_name name of a collection, "*" for all collections
     * @param [in] db_name name of a database, "*" for all databases
     * @return Status operation successfully or not
     */
    virtual Status
    RevokePrivilege(const std::string& role_name, const std::string& privilege, const std::string& collection_name,
                    const std::string& db_name) = 0;

    /**
     * @brief Create a privilege group.
     *
     * @param [in] group_name name of the privilege group
     * @return Status operation successfully or not
     */
    virtual Status
    CreatePrivilegeGroup(const std::string& group_name) = 0;

    /**
     * @brief Drop a privilege group.
     *
     * @param [in] group_name name of the privilege group
     * @return Status operation successfully or not
     */
    virtual Status
    DropPrivilegeGroup(const std::string& group_name) = 0;

    /**
     * @brief List all the privilege groups.
     *
     * @param [out] groups a list of privilege groups
     * @return Status operation successfully or not
     */
    virtual Status
    ListPrivilegeGroups(PrivilegeGroupInfos& groups) = 0;

    /**
     * @brief Add privileges to a privilege group.
     *
     * @param [in] group_name name of the privilege group
     * @param [in] privileges a list of privileges
     * @return Status operation successfully or not
     */
    virtual Status
    AddPrivilegesToGroup(const std::string& group_name, const std::vector<std::string>& privileges) = 0;

    /**
     * @brief Remove privileges from a privilege group.
     *
     * @param [in] group_name name of the privilege group
     * @param [in] privileges a list of privileges
     * @return Status operation successfully or not
     */
    virtual Status
    RemovePrivilegesFromGroup(const std::string& group_name, const std::vector<std::string>& privileges) = 0;
};

using MilvusClientPtr = std::shared_ptr<MilvusClient>;

}  // namespace milvus
