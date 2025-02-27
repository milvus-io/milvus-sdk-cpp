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
#include <memory>
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
        int timeout{0};

        // constructors
        GrpcContextOptions() = default;
        explicit GrpcContextOptions(int timeout_) : timeout{timeout_} {
        }
    };

    MilvusConnection() = default;

    virtual ~MilvusConnection();

    Status
    Connect(const ConnectParam& param);

    Status
    Disconnect();

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
    HasCollection(const proto::milvus::HasCollectionRequest& request, proto::milvus::BoolResponse& response,
                  const GrpcContextOptions& options);

    Status
    GetLoadingProgress(const proto::milvus::GetLoadingProgressRequest& request,
                       proto::milvus::GetLoadingProgressResponse& response, const GrpcContextOptions& options);

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
    AlterCollection(const proto::milvus::AlterCollectionRequest& request, proto::common::Status& response,
                    const GrpcContextOptions& options);

    Status
    AlterCollectionField(const proto::milvus::AlterCollectionFieldRequest& request, proto::common::Status& response,
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
    GetLoadState(const proto::milvus::GetLoadStateRequest& request, proto::milvus::GetLoadStateResponse& response,
                 const GrpcContextOptions& options);

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
    ListAliases(const proto::milvus::ListAliasesRequest& request, proto::milvus::ListAliasesResponse& response,
                const GrpcContextOptions& options);

    Status
    DescribeAlias(const proto::milvus::DescribeAliasRequest& request, proto::milvus::DescribeAliasResponse& response,
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
    DescribeDatabase(const proto::milvus::DescribeDatabaseRequest& request,
                     proto::milvus::DescribeDatabaseResponse& response, const GrpcContextOptions& options);

    Status
    AlterDatabase(const proto::milvus::AlterDatabaseRequest& request, proto::common::Status& response,
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
    HybridSearch(const proto::milvus::HybridSearchRequest& request, proto::milvus::SearchResults& response,
                 const GrpcContextOptions& options);

    Status
    Search(const proto::milvus::SearchRequest& request, proto::milvus::SearchResults& response,
           const GrpcContextOptions& options);

    Status
    Query(const proto::milvus::QueryRequest& request, proto::milvus::QueryResults& response,
          const GrpcContextOptions& options);

    Status
    CalcDistance(const proto::milvus::CalcDistanceRequest& request, proto::milvus::CalcDistanceResults& response,
                 const GrpcContextOptions& options);

    Status
    GetFlushState(const proto::milvus::GetFlushStateRequest& request, proto::milvus::GetFlushStateResponse& response,
                  const GrpcContextOptions& options);

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
    SelectUser(const proto::milvus::SelectUserRequest& request, proto::milvus::SelectUserResponse& response,
               const GrpcContextOptions& options);

    Status
    CreateRole(const proto::milvus::CreateRoleRequest& request, proto::common::Status& response,
               const GrpcContextOptions& options);

    Status
    DropRole(const proto::milvus::DropRoleRequest& request, proto::common::Status& response,
             const GrpcContextOptions& options);

    Status
    OperateUserRole(const proto::milvus::OperateUserRoleRequest& request, proto::common::Status& response,
                    const GrpcContextOptions& options);

    Status
    SelectGrant(const proto::milvus::SelectGrantRequest& request, proto::milvus::SelectGrantResponse& response,
                const GrpcContextOptions& options);

    Status
    SelectRole(const proto::milvus::SelectRoleRequest& request, proto::milvus::SelectRoleResponse& response,
               const GrpcContextOptions& options);

    Status
    OperatePrivilege(const proto::milvus::OperatePrivilegeRequest& request, proto::common::Status& response,
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

    Status
    OperatePrivilegeV2(const proto::milvus::OperatePrivilegeV2Request& request, proto::common::Status& response,
                       const GrpcContextOptions& options);

    Status
    CreateResourceGroup(const proto::milvus::CreateResourceGroupRequest& request, proto::common::Status& response,
                        const GrpcContextOptions& options);

    Status
    DropResourceGroup(const proto::milvus::DropResourceGroupRequest& request, proto::common::Status& response,
                      const GrpcContextOptions& options);

    Status
    DescribeResourceGroup(const proto::milvus::DescribeResourceGroupRequest& request,
                          proto::milvus::DescribeResourceGroupResponse& response, const GrpcContextOptions& options);

    Status
    ListResourceGroups(const proto::milvus::ListResourceGroupsRequest& request,
                       proto::milvus::ListResourceGroupsResponse& response, const GrpcContextOptions& options);

    Status
    UpdateResourceGroups(const proto::milvus::UpdateResourceGroupsRequest& request, proto::common::Status& response,
                         const GrpcContextOptions& options);

    void
    SetHeader(const std::string& key, const std::string& value);

    void
    RemoveHeader(const std::string& key);

    std::string
    GetHeader(const std::string& key) const;

    std::vector<std::pair<std::string, std::string>>
    GetAllHeaders() const;

    const std::string&
    Host() const;

    void
    SetHost(const std::string& host);

    uint16_t
    Port() const;

    void
    SetPort(uint16_t port);

    const std::string&
    Username() const;

    void
    SetUsername(const std::string& username);

    const std::string&
    Password() const;

    void
    SetPassword(const std::string& password);

    const std::string&
    Token() const;

    void
    SetToken(const std::string& token);

    const std::string&
    DbName() const;

    void
    SetDbName(const std::string& db_name);

 private:
    std::unique_ptr<proto::milvus::MilvusService::Stub> stub_;
    std::shared_ptr<grpc::Channel> channel_;
    std::string authorization_value_{};
    std::unordered_map<std::string, std::string> headers_;
    std::string host_;
    uint16_t port_;
    std::string username_;
    std::string password_;
    std::string token_;
    std::string db_name_;

    static Status
    StatusByProtoResponse(const proto::common::Status& status) {
        if (status.code() != proto::common::ErrorCode::Success) {
            return Status{StatusCode::SERVER_FAILED, status.reason()};
        }
        return Status::OK();
    }

    template <typename Response>
    static Status
    StatusByProtoResponse(const Response& response) {
        const auto& status = response.status();
        return StatusByProtoResponse(status);
    }

    static StatusCode
    StatusCodeFromGrpcStatus(const ::grpc::Status& grpc_status) {
        if (grpc_status.error_code() == ::grpc::StatusCode::DEADLINE_EXCEEDED) {
            return StatusCode::TIMEOUT;
        }
        return StatusCode::SERVER_FAILED;
    }

    template <typename Request, typename Response>
    Status
    grpcCall(const char* name,
             grpc::Status (proto::milvus::MilvusService::Stub::*func)(grpc::ClientContext*, const Request&, Response*),
             const Request& request, Response& response, const GrpcContextOptions& options) {
        if (stub_ == nullptr) {
            return {StatusCode::NOT_CONNECTED, "Connection is not ready!"};
        }

        ::grpc::ClientContext context;
        if (options.timeout > 0) {
            auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds{options.timeout};
            context.set_deadline(deadline);
        }

        if (!authorization_value_.empty()) {
            context.AddMetadata("authorization", authorization_value_);
            context.set_authority(authorization_value_);
        }

        ::grpc::Status grpc_status = (stub_.get()->*func)(&context, request, &response);

        if (!grpc_status.ok()) {
            return {StatusCodeFromGrpcStatus(grpc_status), grpc_status.error_message()};
        }

        return StatusByProtoResponse(response);
    }
};
}  // namespace milvus
