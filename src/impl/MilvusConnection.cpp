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

#include "MilvusConnection.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;
using Stub = milvus::proto::milvus::MilvusService::Stub;

namespace milvus {
MilvusConnection::~MilvusConnection() {
    Disconnect();
}

Status
MilvusConnection::Connect(const std::string& uri) {
    ::grpc::ChannelArguments args;
    args.SetMaxSendMessageSize(-1);     // max send message size: 2GB
    args.SetMaxReceiveMessageSize(-1);  // max receive message size: 2GB
    channel_ = ::grpc::CreateCustomChannel(uri, ::grpc::InsecureChannelCredentials(), args);
    if (channel_ != nullptr) {
        stub_ = proto::milvus::MilvusService::NewStub(channel_);
        return Status::OK();
    }

    std::string reason = "Failed to connect uri: " + uri;
    return {StatusCode::NOT_CONNECTED, reason};
}

Status
MilvusConnection::Disconnect() {
    stub_.release();
    channel_.reset();
    return Status::OK();
}

Status
MilvusConnection::CreateCollection(const proto::milvus::CreateCollectionRequest& request,
                                   proto::common::Status& response) {
    return grpcCall("CreateCollection", &Stub::CreateCollection, request, response);
}

Status
MilvusConnection::DropCollection(const proto::milvus::DropCollectionRequest& request, proto::common::Status& response) {
    return Status::OK();
}

Status
MilvusConnection::HasCollection(const proto::milvus::HasCollectionRequest& request,
                                proto::milvus::BoolResponse& response) {
    return grpcCall("HasCollection", &Stub::HasCollection, request, response);
}

Status
MilvusConnection::LoadCollection(const proto::milvus::LoadCollectionRequest& request, proto::common::Status& response) {
    return Status::OK();
}

Status
MilvusConnection::ReleaseCollection(const proto::milvus::ReleaseCollectionRequest& request,
                                    proto::common::Status& response) {
    return Status::OK();
}

Status
MilvusConnection::DescribeCollection(const proto::milvus::DescribeCollectionRequest& request,
                                     proto::milvus::DescribeCollectionResponse& response) {
    return Status::OK();
}

Status
MilvusConnection::GetCollectionStats(const proto::milvus::GetCollectionStatisticsRequest& request,
                                     proto::milvus::GetCollectionStatisticsResponse& response) {
    return Status::OK();
}

Status
MilvusConnection::ShowCollections(const proto::milvus::ShowCollectionsRequest& request,
                                  proto::milvus::ShowCollectionsResponse& response) {
    return grpcCall("ShowCollections", &Stub::ShowCollections, request, response);
}

Status
MilvusConnection::CreatePartition(const proto::milvus::CreatePartitionRequest& request,
                                  proto::common::Status& response) {
    return grpcCall("CreatePartition", &Stub::CreatePartition, request, response);
}

Status
MilvusConnection::DropPartition(const proto::milvus::DropPartitionRequest& request, proto::common::Status& response) {
    return grpcCall("DropPartition", &Stub::DropPartition, request, response);
}

Status
MilvusConnection::HasPartition(const proto::milvus::HasPartitionRequest& request,
                               proto::milvus::BoolResponse& response) {
    return grpcCall("HasParition", &Stub::HasPartition, request, response);
}

Status
MilvusConnection::ShowPartitions(const proto::milvus::ShowPartitionsRequest& request,
                                 proto::milvus::ShowPartitionsResponse& response) {
    return grpcCall("ShowPartitions", &Stub::ShowPartitions, request, response);
}

Status
MilvusConnection::LoadPartitions(const proto::milvus::LoadPartitionsRequest& request, proto::common::Status& response) {
    return grpcCall("LoadPartitions", &Stub::LoadPartitions, request, response);
}

Status
MilvusConnection::ReleasePartitions(const proto::milvus::ReleasePartitionsRequest& request,
                                    proto::common::Status& response) {
    return grpcCall("ReleasePartitions", &Stub::ReleasePartitions, request, response);
}

Status
MilvusConnection::CreateAlias(const proto::milvus::CreateAliasRequest& request, proto::common::Status& response) {
    return grpcCall("HasParition", &Stub::CreateAlias, request, response);
}

Status
MilvusConnection::DropAlias(const proto::milvus::DropAliasRequest& request, proto::common::Status& response) {
    return grpcCall("HasParition", &Stub::DropAlias, request, response);
}

Status
MilvusConnection::AlterAlias(const proto::milvus::AlterAliasRequest& request, proto::common::Status& response) {
    return grpcCall("HasParition", &Stub::AlterAlias, request, response);
}

Status
MilvusConnection::CreateIndex(const proto::milvus::CreateIndexRequest& request, proto::common::Status& response) {
    return grpcCall("CreateIndex", &Stub::CreateIndex, request, response);
}

Status
MilvusConnection::DescribeIndex(const proto::milvus::DescribeIndexRequest& request,
                                proto::milvus::DescribeIndexResponse& response) {
    return Status::OK();
}

Status
MilvusConnection::GetIndexState(const proto::milvus::GetIndexStateRequest& request,
                                proto::milvus::GetIndexStateResponse& response) {
    return grpcCall("HasParition", &Stub::GetIndexState, request, response);
}

Status
MilvusConnection::GetIndexBuildProgress(const proto::milvus::GetIndexBuildProgressRequest& request,
                                        proto::milvus::GetIndexBuildProgressResponse& response) {
    return grpcCall("HasParition", &Stub::GetIndexBuildProgress, request, response);
}

Status
MilvusConnection::DropIndex(const proto::milvus::DropIndexRequest& request, proto::common::Status& response) {
    return grpcCall("HasParition", &Stub::DropIndex, request, response);
}

Status
MilvusConnection::Flush(const proto::milvus::FlushRequest& request, proto::milvus::FlushResponse& response) {
    return grpcCall("Flush", &Stub::Flush, request, response);
}

Status
MilvusConnection::Insert(const proto::milvus::InsertRequest& request, proto::milvus::MutationResult& response) {
    return grpcCall("Insert", &Stub::Insert, request, response);
}

}  // namespace milvus
