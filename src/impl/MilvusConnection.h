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

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "common.pb.h"
#include "milvus.grpc.pb.h"
#include "milvus.pb.h"
#include "milvus/Status.h"
#include "milvus/types/ConnectParam.h"
#include "schema.pb.h"

namespace milvus {

class MilvusConnection {
 public:
    /**
     * options for grpc call
     */
    struct GrpcContextOptions {
        /** timeout in milliseconds */
        uint64_t timeout{0};

        // constructors
        GrpcContextOptions() = default;
        explicit GrpcContextOptions(uint64_t timeout_) : timeout{timeout_} {
        }
    };

    MilvusConnection() = default;

    virtual ~MilvusConnection();

    Status
    Connect(const ConnectParam& param);

    ConnectParam&
    GetConnectParam();

    Status
    Disconnect();

    Status
    UseDatabase(const std::string& db_name);

    Status
    CheckHealth(const proto::milvus::CheckHealthRequest& request, proto::milvus::CheckHealthResponse& response,
                const GrpcContextOptions& options);

    Status
    CreateDatabase(const proto::milvus::CreateDatabaseRequest& request, proto::common::Status& response,
                   const GrpcContextOptions& options);

    Status
    DropDatabase(const proto::milvus::DropDatabaseRequest& request, proto::common::Status& response,
                 const GrpcContextOptions& options);

    Status
    ListDatabases(const proto::milvus::ListDatabasesRequest& request, proto::milvus::ListDatabasesResponse& response,
                  const GrpcContextOptions& options);

    Status
    AlterDatabase(const proto::milvus::AlterDatabaseRequest& request, proto::common::Status& response,
                  const GrpcContextOptions& options);

    Status
    DescribeDatabase(const proto::milvus::DescribeDatabaseRequest& request,
                     proto::milvus::DescribeDatabaseResponse& response, const GrpcContextOptions& options);

    Status
    GetVersion(const proto::milvus::GetVersionRequest& request, proto::milvus::GetVersionResponse& response,
               const GrpcContextOptions& options);
    Status
    CreateCollection(const proto::milvus::CreateCollectionRequest& request, proto::common::Status& response,
                     const GrpcContextOptions& options);

    Status
    DropCollection(const proto::milvus::DropCollectionRequest& request, proto::common::Status& response,
                   const GrpcContextOptions& options);

    Status
    TruncateCollection(const proto::milvus::TruncateCollectionRequest& request,
                       proto::milvus::TruncateCollectionResponse& response, const GrpcContextOptions& options);

    Status
    HasCollection(const proto::milvus::HasCollectionRequest& request, proto::milvus::BoolResponse& response,
                  const GrpcContextOptions& options);

    Status
    LoadCollection(const proto::milvus::LoadCollectionRequest& request, proto::common::Status& response,
                   const GrpcContextOptions& options);

    Status
    ReleaseCollection(const proto::milvus::ReleaseCollectionRequest& request, proto::common::Status& response,
                      const GrpcContextOptions& options);

    Status
    DescribeCollection(const proto::milvus::DescribeCollectionRequest& request,
                       proto::milvus::DescribeCollectionResponse& response, const GrpcContextOptions& options);

    Status
    BatchDescribeCollection(const proto::milvus::BatchDescribeCollectionRequest& request,
                            proto::milvus::BatchDescribeCollectionResponse& response,
                            const GrpcContextOptions& options);

    Status
    GetReplicas(const proto::milvus::GetReplicasRequest& request, proto::milvus::GetReplicasResponse& response,
                const GrpcContextOptions& options);

    Status
    RenameCollection(const proto::milvus::RenameCollectionRequest& request, proto::common::Status& response,
                     const GrpcContextOptions& options);

    Status
    GetCollectionStatistics(const proto::milvus::GetCollectionStatisticsRequest& request,
                            proto::milvus::GetCollectionStatisticsResponse& response,
                            const GrpcContextOptions& options);

    Status
    ShowCollections(const proto::milvus::ShowCollectionsRequest& request,
                    proto::milvus::ShowCollectionsResponse& response, const GrpcContextOptions& options);

    Status
    GetLoadState(const proto::milvus::GetLoadStateRequest& request, proto::milvus::GetLoadStateResponse& response,
                 const GrpcContextOptions& options);

    Status
    GetLoadingProgress(const proto::milvus::GetLoadingProgressRequest,
                       proto::milvus::GetLoadingProgressResponse& response, const GrpcContextOptions& options);

    Status
    AlterCollection(const proto::milvus::AlterCollectionRequest& request, proto::common::Status& response,
                    const GrpcContextOptions& options);

    Status
    AlterCollectionField(const proto::milvus::AlterCollectionFieldRequest& request, proto::common::Status& response,
                         const GrpcContextOptions& options);

    Status
    AddCollectionField(const proto::milvus::AddCollectionFieldRequest& request, proto::common::Status& response,
                       const GrpcContextOptions& options);

    Status
    AddCollectionStructField(const proto::milvus::AddCollectionStructFieldRequest& request,
                             proto::common::Status& response, const GrpcContextOptions& options);

    Status
    AddCollectionFunction(const proto::milvus::AddCollectionFunctionRequest& request, proto::common::Status& response,
                          const GrpcContextOptions& options);
    Status
    AlterCollectionFunction(const proto::milvus::AlterCollectionFunctionRequest& request,
                            proto::common::Status& response, const GrpcContextOptions& options);

    Status
    DropCollectionFunction(const proto::milvus::DropCollectionFunctionRequest& request, proto::common::Status& response,
                           const GrpcContextOptions& options);

    Status
    CreatePartition(const proto::milvus::CreatePartitionRequest& request, proto::common::Status& response,
                    const GrpcContextOptions& options);

    Status
    DropPartition(const proto::milvus::DropPartitionRequest& request, proto::common::Status& response,
                  const GrpcContextOptions& options);

    Status
    HasPartition(const proto::milvus::HasPartitionRequest& request, proto::milvus::BoolResponse& response,
                 const GrpcContextOptions& options);

    Status
    ShowPartitions(const proto::milvus::ShowPartitionsRequest& request, proto::milvus::ShowPartitionsResponse& response,
                   const GrpcContextOptions& options);

    Status
    LoadPartitions(const proto::milvus::LoadPartitionsRequest& request, proto::common::Status& response,
                   const GrpcContextOptions& options);

    Status
    ReleasePartitions(const proto::milvus::ReleasePartitionsRequest& request, proto::common::Status& response,
                      const GrpcContextOptions& options);

    Status
    GetPartitionStatistics(const proto::milvus::GetPartitionStatisticsRequest& request,
                           proto::milvus::GetPartitionStatisticsResponse& response, const GrpcContextOptions& options);

    Status
    CreateAlias(const proto::milvus::CreateAliasRequest& request, proto::common::Status& response,
                const GrpcContextOptions& options);

    Status
    DropAlias(const proto::milvus::DropAliasRequest& request, proto::common::Status& response,
              const GrpcContextOptions& options);

    Status
    AlterAlias(const proto::milvus::AlterAliasRequest& request, proto::common::Status& response,
               const GrpcContextOptions& options);

    Status
    DescribeAlias(const proto::milvus::DescribeAliasRequest& request, proto::milvus::DescribeAliasResponse& response,
                  const GrpcContextOptions& options);

    Status
    ListAliases(const proto::milvus::ListAliasesRequest& request, proto::milvus::ListAliasesResponse& response,
                const GrpcContextOptions& options);

    Status
    CreateIndex(const proto::milvus::CreateIndexRequest& request, proto::common::Status& response,
                const GrpcContextOptions& options);

    Status
    DescribeIndex(const proto::milvus::DescribeIndexRequest& request, proto::milvus::DescribeIndexResponse& response,
                  const GrpcContextOptions& options);

    Status
    GetIndexState(const proto::milvus::GetIndexStateRequest& request, proto::milvus::GetIndexStateResponse& response,
                  const GrpcContextOptions& options);

    Status
    GetIndexBuildProgress(const proto::milvus::GetIndexBuildProgressRequest& request,
                          proto::milvus::GetIndexBuildProgressResponse& response, const GrpcContextOptions& options);

    Status
    DropIndex(const proto::milvus::DropIndexRequest& request, proto::common::Status& response,
              const GrpcContextOptions& options);

    Status
    AlterIndex(const proto::milvus::AlterIndexRequest& request, proto::common::Status& response,
               const GrpcContextOptions& options);

    Status
    Flush(const proto::milvus::FlushRequest& request, proto::milvus::FlushResponse& response,
          const GrpcContextOptions& options);

    Status
    Insert(const proto::milvus::InsertRequest& request, proto::milvus::MutationResult& response,
           const GrpcContextOptions& options);

    Status
    Upsert(const proto::milvus::UpsertRequest& request, proto::milvus::MutationResult& response,
           const GrpcContextOptions& options);

    Status
    Delete(const proto::milvus::DeleteRequest& request, proto::milvus::MutationResult& response,
           const GrpcContextOptions& options);

    Status
    Search(const proto::milvus::SearchRequest& request, proto::milvus::SearchResults& response,
           const GrpcContextOptions& options);

    Status
    HybridSearch(const proto::milvus::HybridSearchRequest& request, proto::milvus::SearchResults& response,
                 const GrpcContextOptions& options);

    Status
    Query(const proto::milvus::QueryRequest& request, proto::milvus::QueryResults& response,
          const GrpcContextOptions& options);

    Status
    RunAnalyzer(const proto::milvus::RunAnalyzerRequest& request, proto::milvus::RunAnalyzerResponse& response,
                const GrpcContextOptions& options);

    Status
    GetFlushState(const proto::milvus::GetFlushStateRequest& request, proto::milvus::GetFlushStateResponse& response,
                  const GrpcContextOptions& options);

    Status
    FlushAll(const proto::milvus::FlushAllRequest& request, proto::milvus::FlushAllResponse& response,
             const GrpcContextOptions& options);

    Status
    GetFlushAllState(const proto::milvus::GetFlushAllStateRequest& request,
                     proto::milvus::GetFlushAllStateResponse& response, const GrpcContextOptions& options);

    Status
    RefreshExternalCollection(const proto::milvus::RefreshExternalCollectionRequest& request,
                              proto::milvus::RefreshExternalCollectionResponse& response,
                              const GrpcContextOptions& options);

    Status
    GetRefreshExternalCollectionProgress(const proto::milvus::GetRefreshExternalCollectionProgressRequest& request,
                                         proto::milvus::GetRefreshExternalCollectionProgressResponse& response,
                                         const GrpcContextOptions& options);

    Status
    ListRefreshExternalCollectionJobs(const proto::milvus::ListRefreshExternalCollectionJobsRequest& request,
                                      proto::milvus::ListRefreshExternalCollectionJobsResponse& response,
                                      const GrpcContextOptions& options);

    Status
    AddFileResource(const proto::milvus::AddFileResourceRequest& request, proto::common::Status& response,
                    const GrpcContextOptions& options);

    Status
    RemoveFileResource(const proto::milvus::RemoveFileResourceRequest& request, proto::common::Status& response,
                       const GrpcContextOptions& options);

    Status
    ListFileResources(const proto::milvus::ListFileResourcesRequest& request,
                      proto::milvus::ListFileResourcesResponse& response, const GrpcContextOptions& options);

    Status
    GetPersistentSegmentInfo(const proto::milvus::GetPersistentSegmentInfoRequest& request,
                             proto::milvus::GetPersistentSegmentInfoResponse& response,
                             const GrpcContextOptions& options);

    Status
    GetQuerySegmentInfo(const proto::milvus::GetQuerySegmentInfoRequest& request,
                        proto::milvus::GetQuerySegmentInfoResponse& response, const GrpcContextOptions& options);

    Status
    GetMetrics(const proto::milvus::GetMetricsRequest& request, proto::milvus::GetMetricsResponse& response,
               const GrpcContextOptions& options);

    Status
    LoadBalance(const proto::milvus::LoadBalanceRequest& request, proto::common::Status& response,
                const GrpcContextOptions& options);

    Status
    GetCompactionState(const proto::milvus::GetCompactionStateRequest& request,
                       proto::milvus::GetCompactionStateResponse& response, const GrpcContextOptions& options);

    Status
    ManualCompaction(const proto::milvus::ManualCompactionRequest& request,
                     proto::milvus::ManualCompactionResponse& response, const GrpcContextOptions& options);

    Status
    GetCompactionPlans(const proto::milvus::GetCompactionPlansRequest& request,
                       proto::milvus::GetCompactionPlansResponse& response, const GrpcContextOptions& options);

    Status
    CreateSnapshot(const proto::milvus::CreateSnapshotRequest& request, proto::common::Status& response,
                   const GrpcContextOptions& options);

    Status
    DropSnapshot(const proto::milvus::DropSnapshotRequest& request, proto::common::Status& response,
                 const GrpcContextOptions& options);

    Status
    ListSnapshots(const proto::milvus::ListSnapshotsRequest& request, proto::milvus::ListSnapshotsResponse& response,
                  const GrpcContextOptions& options);

    Status
    DescribeSnapshot(const proto::milvus::DescribeSnapshotRequest& request,
                     proto::milvus::DescribeSnapshotResponse& response, const GrpcContextOptions& options);

    Status
    RestoreSnapshot(const proto::milvus::RestoreSnapshotRequest& request,
                    proto::milvus::RestoreSnapshotResponse& response, const GrpcContextOptions& options);

    Status
    GetRestoreSnapshotState(const proto::milvus::GetRestoreSnapshotStateRequest& request,
                            proto::milvus::GetRestoreSnapshotStateResponse& response,
                            const GrpcContextOptions& options);

    Status
    ListRestoreSnapshotJobs(const proto::milvus::ListRestoreSnapshotJobsRequest& request,
                            proto::milvus::ListRestoreSnapshotJobsResponse& response,
                            const GrpcContextOptions& options);

    Status
    PinSnapshotData(const proto::milvus::PinSnapshotDataRequest& request,
                    proto::milvus::PinSnapshotDataResponse& response, const GrpcContextOptions& options);

    Status
    UnpinSnapshotData(const proto::milvus::UnpinSnapshotDataRequest& request, proto::common::Status& response,
                      const GrpcContextOptions& options);

    Status
    CreateCredential(const proto::milvus::CreateCredentialRequest& request, proto::common::Status& response,
                     const GrpcContextOptions& options);

    Status
    UpdateCredential(const proto::milvus::UpdateCredentialRequest& request, proto::common::Status& response,
                     const GrpcContextOptions& options);

    Status
    DeleteCredential(const proto::milvus::DeleteCredentialRequest& request, proto::common::Status& response,
                     const GrpcContextOptions& options);

    Status
    ListCredUsers(const proto::milvus::ListCredUsersRequest& request, proto::milvus::ListCredUsersResponse& response,
                  const GrpcContextOptions& options);

    Status
    CreateResourceGroup(const proto::milvus::CreateResourceGroupRequest& request, proto::common::Status& response,
                        const GrpcContextOptions& options);

    Status
    DropResourceGroup(const proto::milvus::DropResourceGroupRequest& request, proto::common::Status& response,
                      const GrpcContextOptions& options);
    Status
    UpdateResourceGroups(const proto::milvus::UpdateResourceGroupsRequest& request, proto::common::Status& response,
                         const GrpcContextOptions& options);

    Status
    TransferNode(const proto::milvus::TransferNodeRequest& request, proto::common::Status& response,
                 const GrpcContextOptions& options);

    Status
    TransferReplica(const proto::milvus::TransferReplicaRequest& request, proto::common::Status& response,
                    const GrpcContextOptions& options);

    Status
    ListResourceGroups(const proto::milvus::ListResourceGroupsRequest& request,
                       proto::milvus::ListResourceGroupsResponse& response, const GrpcContextOptions& options);

    Status
    DescribeResourceGroup(const proto::milvus::DescribeResourceGroupRequest& request,
                          proto::milvus::DescribeResourceGroupResponse& response, const GrpcContextOptions& options);

    Status
    GetReplicateConfiguration(const proto::milvus::GetReplicateConfigurationRequest& request,
                              proto::milvus::GetReplicateConfigurationResponse& response,
                              const GrpcContextOptions& options);

    Status
    UpdateReplicateConfiguration(const proto::milvus::UpdateReplicateConfigurationRequest& request,
                                 proto::common::Status& response, const GrpcContextOptions& options);

    Status
    GetReplicateInfo(const proto::milvus::GetReplicateInfoRequest& request,
                     proto::milvus::GetReplicateInfoResponse& response, const GrpcContextOptions& options);

    Status
    DumpMessages(const proto::milvus::DumpMessagesRequest& request, const GrpcContextOptions& options,
                 const std::function<Status(const proto::common::ImmutableMessage&)>& on_message);

    Status
    SelectUser(const proto::milvus::SelectUserRequest& request, proto::milvus::SelectUserResponse& response,
               const GrpcContextOptions& options);

    Status
    SelectRole(const proto::milvus::SelectRoleRequest& request, proto::milvus::SelectRoleResponse& response,
               const GrpcContextOptions& options);

    Status
    SelectGrant(const proto::milvus::SelectGrantRequest& request, proto::milvus::SelectGrantResponse& response,
                const GrpcContextOptions& options);

    Status
    CreateRole(const proto::milvus::CreateRoleRequest& request, proto::common::Status& response,
               const GrpcContextOptions& options);

    Status
    AlterRole(const proto::milvus::AlterRoleRequest& request, proto::common::Status& response,
              const GrpcContextOptions& options);

    Status
    DropRole(const proto::milvus::DropRoleRequest& request, proto::common::Status& response,
             const GrpcContextOptions& options);

    Status
    OperateUserRole(const proto::milvus::OperateUserRoleRequest& request, proto::common::Status& response,
                    const GrpcContextOptions& options);

    Status
    OperatePrivilegeV2(const proto::milvus::OperatePrivilegeV2Request& request, proto::common::Status& response,
                       const GrpcContextOptions& options);

    Status
    CreatePrivilegeGroup(const proto::milvus::CreatePrivilegeGroupRequest& request, proto::common::Status& response,
                         const GrpcContextOptions& options);

    Status
    DropPrivilegeGroup(const proto::milvus::DropPrivilegeGroupRequest& request, proto::common::Status& response,
                       const GrpcContextOptions& options);

    Status
    ListPrivilegeGroups(const proto::milvus::ListPrivilegeGroupsRequest& request,
                        proto::milvus::ListPrivilegeGroupsResponse& response, const GrpcContextOptions& options);

    Status
    OperatePrivilegeGroup(const proto::milvus::OperatePrivilegeGroupRequest& request, proto::common::Status& response,
                          const GrpcContextOptions& options);

 private:
    std::mutex stub_mtx_;
    std::shared_ptr<proto::milvus::MilvusService::Stub> stub_;
    std::shared_ptr<grpc::Channel> channel_;
    ConnectParam param_;

    static Status
    StatusByProtoResponse(const proto::common::Status& status);

    static Status
    StatusByProtoResponse(const proto::milvus::GetReplicateInfoResponse& response);

    template <typename Response>
    static Status
    StatusByProtoResponse(const Response& response);

    static Status
    StatusCodeFromGrpcStatus(const ::grpc::Status& grpc_status);

    template <typename Request, typename Response>
    Status
    grpcCall(const char* name,
             grpc::Status (proto::milvus::MilvusService::Stub::*func)(grpc::ClientContext*, const Request&, Response*),
             const Request& request, Response& response, const GrpcContextOptions& options) {
        std::shared_ptr<proto::milvus::MilvusService::Stub> stub;
        {
            std::lock_guard<std::mutex> lock(stub_mtx_);
            stub = stub_;
        }
        if (stub == nullptr) {
            return {StatusCode::NOT_CONNECTED, "Connection is not ready!"};
        }

        ::grpc::ClientContext context;
        if (options.timeout > 0) {
            auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds{options.timeout};
            context.set_deadline(deadline);
        }

        ::grpc::Status grpc_status = (stub.get()->*func)(&context, request, &response);

        // TODO: check the error codes and do retry here
        // The following grpc error codes cannot be retried:
        //   grpc::StatusCode::DEADLINE_EXCEEDED
        //   grpc::StatusCode::PERMISSION_DENIED
        //   grpc::StatusCode::UNAUTHENTICATED
        //   grpc::StatusCode::INVALID_ARGUMENT
        //   grpc::StatusCode::LREADY_EXISTS
        //   grpc::StatusCode::RESOURCE_EXHAUSTED
        //   grpc::StatusCode::UNIMPLEMENTED
        if (!grpc_status.ok()) {
            return StatusCodeFromGrpcStatus(grpc_status);
        }

        // Some milvus error codes can be retried:
        //   response.status().error_code() == io.milvus.grpc.ErrorCode.RateLimit
        //   or response.status()code() == 8 can be retried
        return StatusByProtoResponse(response);
    }
};

using GrpcOpts = MilvusConnection::GrpcContextOptions;
using MilvusConnectionPtr = std::shared_ptr<MilvusConnection>;

}  // namespace milvus
