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

namespace milvus {
class MilvusMockedService : public milvus::proto::milvus::MilvusService::Service::Service {
 public:
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
    MOCK_METHOD3(GetCollectionStatistics,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::GetCollectionStatisticsRequest*,
                                ::milvus::proto::milvus::GetCollectionStatisticsResponse*));
    MOCK_METHOD3(ShowCollections,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::ShowCollectionsRequest*,
                                ::milvus::proto::milvus::ShowCollectionsResponse*));
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
    MOCK_METHOD3(Insert, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::InsertRequest*,
                                        ::milvus::proto::milvus::MutationResult*));
    MOCK_METHOD3(Delete, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::DeleteRequest*,
                                        ::milvus::proto::milvus::MutationResult*));
    MOCK_METHOD3(Search, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::SearchRequest*,
                                        ::milvus::proto::milvus::SearchResults*));
    MOCK_METHOD3(Flush, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::FlushRequest*,
                                       ::milvus::proto::milvus::FlushResponse*));
    MOCK_METHOD3(Query, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::QueryRequest*,
                                       ::milvus::proto::milvus::QueryResults*));
    MOCK_METHOD3(GetFlushState,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::GetFlushStateRequest*,
                                ::milvus::proto::milvus::GetFlushStateResponse*));
    MOCK_METHOD3(CalcDistance,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::CalcDistanceRequest*,
                                ::milvus::proto::milvus::CalcDistanceResults*));
    MOCK_METHOD3(GetPersistentSegmentInfo,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::GetPersistentSegmentInfoRequest*,
                                ::milvus::proto::milvus::GetPersistentSegmentInfoResponse*));
    MOCK_METHOD3(GetQuerySegmentInfo,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::GetQuerySegmentInfoRequest*,
                                ::milvus::proto::milvus::GetQuerySegmentInfoResponse*));
    MOCK_METHOD3(Dummy, ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::DummyRequest*,
                                       ::milvus::proto::milvus::DummyResponse*));
    MOCK_METHOD3(RegisterLink,
                 ::grpc::Status(::grpc::ServerContext*, const ::milvus::proto::milvus::RegisterLinkRequest*,
                                ::milvus::proto::milvus::RegisterLinkResponse*));
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
};

}  // namespace milvus