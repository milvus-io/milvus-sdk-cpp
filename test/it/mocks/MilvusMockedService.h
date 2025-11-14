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
#include <gmock/gmock.h>

#include "milvus.grpc.pb.h"
#include "milvus.pb.h"

namespace milvus {
class MilvusMockedService : public ::milvus::proto::milvus::MilvusService::Service {
 public:
    MOCK_METHOD3(GetVersion, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::GetVersionRequest*,
                                            ::milvus::proto::milvus::GetVersionResponse*));

    MOCK_METHOD3(Connect, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::ConnectRequest*,
                                         ::milvus::proto::milvus::ConnectResponse*));

    MOCK_METHOD3(CreateDatabase,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::CreateDatabaseRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(DropDatabase,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::DropDatabaseRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(ListDatabases,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::ListDatabasesRequest*,
                                ::milvus::proto::milvus::ListDatabasesResponse*));

    MOCK_METHOD3(AlterDatabase,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::AlterDatabaseRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(DescribeDatabase,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::DescribeDatabaseRequest*,
                                ::milvus::proto::milvus::DescribeDatabaseResponse*));

    MOCK_METHOD3(CreateCollection,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::CreateCollectionRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(DropCollection,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::DropCollectionRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(HasCollection,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::HasCollectionRequest*,
                                ::milvus::proto::milvus::BoolResponse*));

    MOCK_METHOD3(LoadCollection,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::LoadCollectionRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(ReleaseCollection,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::ReleaseCollectionRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(DescribeCollection,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::DescribeCollectionRequest*,
                                ::milvus::proto::milvus::DescribeCollectionResponse*));

    MOCK_METHOD3(RenameCollection,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::RenameCollectionRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(GetCollectionStatistics,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::GetCollectionStatisticsRequest*,
                                ::milvus::proto::milvus::GetCollectionStatisticsResponse*));

    MOCK_METHOD3(ShowCollections,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::ShowCollectionsRequest*,
                                ::milvus::proto::milvus::ShowCollectionsResponse*));

    MOCK_METHOD3(GetLoadState,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::GetLoadStateRequest*,
                                ::milvus::proto::milvus::GetLoadStateResponse*));

    MOCK_METHOD3(GetLoadingProgress,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::GetLoadingProgressRequest*,
                                ::milvus::proto::milvus::GetLoadingProgressResponse*));

    MOCK_METHOD3(AlterCollection,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::AlterCollectionRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(AlterCollectionField,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::AlterCollectionFieldRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(CreatePartition,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::CreatePartitionRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(DropPartition,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::DropPartitionRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(HasPartition,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::HasPartitionRequest*,
                                ::milvus::proto::milvus::BoolResponse*));

    MOCK_METHOD3(LoadPartitions,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::LoadPartitionsRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(ReleasePartitions,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::ReleasePartitionsRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(GetPartitionStatistics,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::GetPartitionStatisticsRequest*,
                                ::milvus::proto::milvus::GetPartitionStatisticsResponse*));

    MOCK_METHOD3(ShowPartitions,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::ShowPartitionsRequest*,
                                ::milvus::proto::milvus::ShowPartitionsResponse*));

    MOCK_METHOD3(CreateAlias, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::CreateAliasRequest*,
                                             ::milvus::proto::common::Status*));

    MOCK_METHOD3(DropAlias, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::DropAliasRequest*,
                                           ::milvus::proto::common::Status*));

    MOCK_METHOD3(AlterAlias, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::AlterAliasRequest*,
                                            ::milvus::proto::common::Status*));

    MOCK_METHOD3(DescribeAlias,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::DescribeAliasRequest*,
                                ::milvus::proto::milvus::DescribeAliasResponse*));

    MOCK_METHOD3(ListAliases, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::ListAliasesRequest*,
                                             ::milvus::proto::milvus::ListAliasesResponse*));

    MOCK_METHOD3(CreateIndex, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::CreateIndexRequest*,
                                             ::milvus::proto::common::Status*));

    MOCK_METHOD3(DescribeIndex,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::DescribeIndexRequest*,
                                ::milvus::proto::milvus::DescribeIndexResponse*));

    MOCK_METHOD3(GetIndexState,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::GetIndexStateRequest*,
                                ::milvus::proto::milvus::GetIndexStateResponse*));

    MOCK_METHOD3(GetIndexBuildProgress,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::GetIndexBuildProgressRequest*,
                                ::milvus::proto::milvus::GetIndexBuildProgressResponse*));

    MOCK_METHOD3(DropIndex, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::DropIndexRequest*,
                                           ::milvus::proto::common::Status*));

    MOCK_METHOD3(AlterIndex, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::AlterIndexRequest*,
                                            ::milvus::proto::common::Status*));

    MOCK_METHOD3(Insert, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::InsertRequest*,
                                        ::milvus::proto::milvus::MutationResult*));

    MOCK_METHOD3(Upsert, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::UpsertRequest*,
                                        ::milvus::proto::milvus::MutationResult*));

    MOCK_METHOD3(Delete, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::DeleteRequest*,
                                        ::milvus::proto::milvus::MutationResult*));

    MOCK_METHOD3(Search, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::SearchRequest*,
                                        ::milvus::proto::milvus::SearchResults*));

    MOCK_METHOD3(HybridSearch,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::HybridSearchRequest*,
                                ::milvus::proto::milvus::SearchResults*));

    MOCK_METHOD3(Flush, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::FlushRequest*,
                                       ::milvus::proto::milvus::FlushResponse*));

    MOCK_METHOD3(Query, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::QueryRequest*,
                                       ::milvus::proto::milvus::QueryResults*));

    MOCK_METHOD3(RunAnalyzer, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::RunAnalyzerRequest*,
                                             ::milvus::proto::milvus::RunAnalyzerResponse*));

    MOCK_METHOD3(GetFlushState,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::GetFlushStateRequest*,
                                ::milvus::proto::milvus::GetFlushStateResponse*));

    MOCK_METHOD3(GetPersistentSegmentInfo,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::GetPersistentSegmentInfoRequest*,
                                ::milvus::proto::milvus::GetPersistentSegmentInfoResponse*));

    MOCK_METHOD3(GetQuerySegmentInfo,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::GetQuerySegmentInfoRequest*,
                                ::milvus::proto::milvus::GetQuerySegmentInfoResponse*));

    MOCK_METHOD3(GetMetrics, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::GetMetricsRequest*,
                                            ::milvus::proto::milvus::GetMetricsResponse*));

    MOCK_METHOD3(LoadBalance, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::LoadBalanceRequest*,
                                             ::milvus::proto::common::Status*));

    MOCK_METHOD3(GetCompactionState,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::GetCompactionStateRequest*,
                                ::milvus::proto::milvus::GetCompactionStateResponse*));

    MOCK_METHOD3(ManualCompaction,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::ManualCompactionRequest*,
                                ::milvus::proto::milvus::ManualCompactionResponse*));

    MOCK_METHOD3(GetCompactionStateWithPlans,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::GetCompactionPlansRequest*,
                                ::milvus::proto::milvus::GetCompactionPlansResponse*));

    MOCK_METHOD3(CreateCredential,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::CreateCredentialRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(UpdateCredential,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::UpdateCredentialRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(DeleteCredential,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::DeleteCredentialRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(ListCredUsers,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::ListCredUsersRequest*,
                                ::milvus::proto::milvus::ListCredUsersResponse*));

    MOCK_METHOD3(CreateResourceGroup,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::CreateResourceGroupRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(DropResourceGroup,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::DropResourceGroupRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(UpdateResourceGroups,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::UpdateResourceGroupsRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(TransferNode,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::TransferNodeRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(TransferReplica,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::TransferReplicaRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(ListResourceGroups,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::ListResourceGroupsRequest*,
                                ::milvus::proto::milvus::ListResourceGroupsResponse*));

    MOCK_METHOD3(DescribeResourceGroup,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::DescribeResourceGroupRequest*,
                                ::milvus::proto::milvus::DescribeResourceGroupResponse*));

    MOCK_METHOD3(SelectUser, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::SelectUserRequest*,
                                            ::milvus::proto::milvus::SelectUserResponse*));

    MOCK_METHOD3(SelectRole, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::SelectRoleRequest*,
                                            ::milvus::proto::milvus::SelectRoleResponse*));

    MOCK_METHOD3(SelectGrant, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::SelectGrantRequest*,
                                             ::milvus::proto::milvus::SelectGrantResponse*));

    MOCK_METHOD3(CreateRole, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::CreateRoleRequest*,
                                            ::milvus::proto::common::Status*));

    MOCK_METHOD3(DropRole, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::DropRoleRequest*,
                                          ::milvus::proto::common::Status*));

    MOCK_METHOD3(OperateUserRole,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::OperateUserRoleRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(OperatePrivilegeV2,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::OperatePrivilegeV2Request*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(CreatePrivilegeGroup,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::CreatePrivilegeGroupRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(DropPrivilegeGroup,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::DropPrivilegeGroupRequest*,
                                ::milvus::proto::common::Status*));

    MOCK_METHOD3(ListPrivilegeGroups,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::ListPrivilegeGroupsRequest*,
                                ::milvus::proto::milvus::ListPrivilegeGroupsResponse*));

    MOCK_METHOD3(OperatePrivilegeGroup,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::OperatePrivilegeGroupRequest*,
                                ::milvus::proto::common::Status*));
};

}  // namespace milvus