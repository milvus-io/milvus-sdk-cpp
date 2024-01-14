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

#include "MilvusConnection.h"
#include "milvus/MilvusClient.h"

/**
 *  @brief namespace milvus
 */
namespace milvus {

class MilvusClientImpl : public MilvusClient {
 public:
    MilvusClientImpl() = default;
    virtual ~MilvusClientImpl();

    Status
    Connect(const ConnectParam& connect_param) final;

    Status
    Disconnect() final;

    Status
    GetVersion(std::string& version) final;

    Status
    CreateCollection(const CollectionSchema& schema) final;

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
    CreateAlias(const std::string& collection_name, const std::string& alias) final;

    Status
    DropAlias(const std::string& alias) final;

    Status
    AlterAlias(const std::string& collection_name, const std::string& alias) final;

    Status
    CreateIndex(const std::string& collection_name, const IndexDesc& index_desc,
                const ProgressMonitor& progress_monitor) final;

    Status
    DescribeIndex(const std::string& collection_name, const std::string& field_name, IndexDesc& index_desc) final;

    Status
    GetIndexState(const std::string& collection_name, const std::string& field_name, IndexState& state) final;

    Status
    GetIndexBuildProgress(const std::string& collection_name, const std::string& field_name,
                          IndexProgress& progress) final;

    Status
    DropIndex(const std::string& collection_name, const std::string& field_name) final;

    Status
    Insert(const std::string& collection_name, const std::string& partition_name,
           const std::vector<FieldDataPtr>& fields, DmlResults& results) final;

    Status
    Delete(const std::string& collection_name, const std::string& partition_name, const std::string& expression,
           DmlResults& results) final;

    Status
    Search(const SearchArguments& arguments, SearchResults& results, int timeout) final;

    Status
    Query(const QueryArguments& arguments, QueryResults& results, int timeout) final;

    Status
    CalcDistance(const CalcDistanceArguments& arguments, DistanceArray& results) final;

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

 private:
    using GrpcOpts = MilvusConnection::GrpcContextOptions;

    /**
     * Internal wait for status query done.
     *
     * @param [in] query_function one time query for return Status, return TIMEOUT status if not done
     * @param [in] progress_monitor timeout setting for waiting progress
     * @return Status, the final status
     */
    static Status
    WaitForStatus(const std::function<Status(Progress&)>& query_function, const ProgressMonitor& progress_monitor);

    /**
     * @brief template for public api call
     *        validate -> pre -> rpc -> wait_for_status -> post
     */
    template <typename Request, typename Response>
    Status
    apiHandler(const std::function<Status(void)>& validate, std::function<Request(void)> pre,
               Status (MilvusConnection::*rpc)(const Request&, Response&, const GrpcOpts&),
               std::function<Status(const Response&)> wait_for_status, std::function<void(const Response&)> post,
               const GrpcOpts& options = GrpcOpts{}) {
        if (connection_ == nullptr) {
            return {StatusCode::NOT_CONNECTED, "Connection is not ready!"};
        }

        if (validate) {
            auto status = validate();
            if (!status.IsOk()) {
                return status;
            }
        }

        Request rpc_request = pre();
        Response rpc_response;
        auto status = std::bind(rpc, connection_.get(), std::placeholders::_1, std::placeholders::_2,
                                std::placeholders::_3)(rpc_request, rpc_response, options);
        if (!status.IsOk()) {
            // response's status already checked in connection class
            return status;
        }

        if (wait_for_status) {
            status = wait_for_status(rpc_response);
        }

        if (status.IsOk() && post) {
            post(rpc_response);
        }
        return status;
    }

    /**
     * @brief template for public api call
     */
    template <typename Request, typename Response>
    Status
    apiHandler(std::function<Status(void)> validate, std::function<Request(void)> pre,
               Status (MilvusConnection::*rpc)(const Request&, Response&, const GrpcOpts&),
               std::function<void(const Response&)> post, const GrpcOpts& options = GrpcOpts{}) {
        return apiHandler(validate, pre, rpc, std::function<Status(const Response&)>{}, post, options);
    }

    /**
     * @brief template for public api call
     */
    template <typename Request, typename Response>
    Status
    apiHandler(std::function<Status(void)> validate, std::function<Request(void)> pre,
               Status (MilvusConnection::*rpc)(const Request&, Response&, const GrpcOpts&),
               const GrpcOpts& options = GrpcOpts{}) {
        return apiHandler(validate, pre, rpc, std::function<Status(const Response&)>{},
                          std::function<void(const Response&)>{}, options);
    }

    /**
     * @brief template for public api call
     */
    template <typename Request, typename Response>
    Status
    apiHandler(std::function<Request(void)> pre,
               Status (MilvusConnection::*rpc)(const Request&, Response&, const GrpcOpts&),
               std::function<void(const Response&)> post, const GrpcOpts& options = GrpcOpts{}) {
        return apiHandler(std::function<Status(void)>{}, pre, rpc, std::function<Status(const Response&)>{}, post,
                          options);
    }

    /**
     * @brief template for public api call
     */
    template <typename Request, typename Response>
    Status
    apiHandler(std::function<Request(void)> pre,
               Status (MilvusConnection::*rpc)(const Request&, Response&, const GrpcOpts&),
               const GrpcOpts& options = GrpcOpts{}) {
        return apiHandler(std::function<Status(void)>{}, pre, rpc, std::function<Status(const Response&)>{},
                          std::function<void(const Response&)>{}, options);
    }

 private:
    std::shared_ptr<MilvusConnection> connection_;
};

}  // namespace milvus
