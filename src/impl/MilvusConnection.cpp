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

#include <fstream>
#include <memory>

#include "grpcpp/security/credentials.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;
using Stub = milvus::proto::milvus::MilvusService::Stub;

namespace {

std::shared_ptr<grpc::ChannelCredentials>
createTlsCredentials(const std::string& cert, const std::string& key, const std::string& ca_cert) {
    auto read_contents = [](const std::string& filename) -> std::string {
        if (filename.empty()) {
            return "";
        }
        std::ifstream fs;
        fs.open(filename);
        if (fs) {
            fs.seekg(0, std::ios::end);
            auto size = fs.tellg();
            std::string buffer(size, '\0');
            fs.seekg(0);
            fs.read(&buffer[0], size);
            return std::move(buffer);
        }
        return "";
    };
    grpc::SslCredentialsOptions opt{read_contents(ca_cert), read_contents(key), read_contents(cert)};
    return ::grpc::SslCredentials(opt);
}
}  // namespace

namespace milvus {
MilvusConnection::~MilvusConnection() {
    Disconnect();
}

Status
MilvusConnection::Connect(const ConnectParam& param) {
    authorization_value_ = param.Authorizations();
    std::shared_ptr<grpc::ChannelCredentials> credentials{nullptr};
    auto uri = param.Uri();

    ::grpc::ChannelArguments args;
    args.SetMaxSendMessageSize(-1);     // max send message size: 2GB
    args.SetMaxReceiveMessageSize(-1);  // max receive message size: 2GB

    if (param.TlsEnabled()) {
        if (!param.ServerName().empty()) {
            args.SetSslTargetNameOverride(param.ServerName());
        }
        credentials = createTlsCredentials(param.Cert(), param.Key(), param.CaCert());
    } else {
        credentials = ::grpc::InsecureChannelCredentials();
    }

    channel_ = ::grpc::CreateCustomChannel(uri, credentials, args);
    auto connected = channel_->WaitForConnected(std::chrono::system_clock::now() +
                                                std::chrono::milliseconds{param.ConnectTimeout()});
    if (connected) {
        stub_ = proto::milvus::MilvusService::NewStub(channel_);
        return Status::OK();
    }

    std::string reason = "Failed to connect uri: " + uri;
    return {StatusCode::NOT_CONNECTED, reason};
}

Status
MilvusConnection::Disconnect() {
    stub_.reset();
    channel_.reset();
    return Status::OK();
}

Status
MilvusConnection::GetVersion(const proto::milvus::GetVersionRequest& request,
                             proto::milvus::GetVersionResponse& response, const GrpcContextOptions& options) {
    return grpcCall("GetVersion", &Stub::GetVersion, request, response, options);
}

Status
MilvusConnection::CreateCollection(const proto::milvus::CreateCollectionRequest& request,
                                   proto::common::Status& response, const GrpcContextOptions& options) {
    return grpcCall("CreateCollection", &Stub::CreateCollection, request, response, options);
}

Status
MilvusConnection::DropCollection(const proto::milvus::DropCollectionRequest& request, proto::common::Status& response,
                                 const GrpcContextOptions& options) {
    return grpcCall("DropCollection", &Stub::DropCollection, request, response, options);
}

Status
MilvusConnection::HasCollection(const proto::milvus::HasCollectionRequest& request,
                                proto::milvus::BoolResponse& response, const GrpcContextOptions& options) {
    return grpcCall("HasCollection", &Stub::HasCollection, request, response, options);
}

Status
MilvusConnection::LoadCollection(const proto::milvus::LoadCollectionRequest& request, proto::common::Status& response,
                                 const GrpcContextOptions& options) {
    return grpcCall("LoadCollection", &Stub::LoadCollection, request, response, options);
}

Status
MilvusConnection::ReleaseCollection(const proto::milvus::ReleaseCollectionRequest& request,
                                    proto::common::Status& response, const GrpcContextOptions& options) {
    return grpcCall("ReleaseCollection", &Stub::ReleaseCollection, request, response, options);
}

Status
MilvusConnection::DescribeCollection(const proto::milvus::DescribeCollectionRequest& request,
                                     proto::milvus::DescribeCollectionResponse& response,
                                     const GrpcContextOptions& options) {
    return grpcCall("DescribeCollection", &Stub::DescribeCollection, request, response, options);
}

Status
MilvusConnection::RenameCollection(const proto::milvus::RenameCollectionRequest& request,
                                   proto::common::Status& response, const GrpcContextOptions& options) {
    return grpcCall("RenameCollection", &Stub::RenameCollection, request, response, options);
}

Status
MilvusConnection::GetCollectionStatistics(const proto::milvus::GetCollectionStatisticsRequest& request,
                                          proto::milvus::GetCollectionStatisticsResponse& response,
                                          const GrpcContextOptions& options) {
    return grpcCall("GetCollectionStatistics", &Stub::GetCollectionStatistics, request, response, options);
}

Status
MilvusConnection::ShowCollections(const proto::milvus::ShowCollectionsRequest& request,
                                  proto::milvus::ShowCollectionsResponse& response, const GrpcContextOptions& options) {
    return grpcCall("ShowCollections", &Stub::ShowCollections, request, response, options);
}

Status
MilvusConnection::CreatePartition(const proto::milvus::CreatePartitionRequest& request, proto::common::Status& response,
                                  const GrpcContextOptions& options) {
    return grpcCall("CreatePartition", &Stub::CreatePartition, request, response, options);
}

Status
MilvusConnection::DropPartition(const proto::milvus::DropPartitionRequest& request, proto::common::Status& response,
                                const GrpcContextOptions& options) {
    return grpcCall("DropPartition", &Stub::DropPartition, request, response, options);
}

Status
MilvusConnection::HasPartition(const proto::milvus::HasPartitionRequest& request, proto::milvus::BoolResponse& response,
                               const GrpcContextOptions& options) {
    return grpcCall("HasPartition", &Stub::HasPartition, request, response, options);
}

Status
MilvusConnection::ShowPartitions(const proto::milvus::ShowPartitionsRequest& request,
                                 proto::milvus::ShowPartitionsResponse& response, const GrpcContextOptions& options) {
    return grpcCall("ShowPartitions", &Stub::ShowPartitions, request, response, options);
}

Status
MilvusConnection::LoadPartitions(const proto::milvus::LoadPartitionsRequest& request, proto::common::Status& response,
                                 const GrpcContextOptions& options) {
    return grpcCall("LoadPartitions", &Stub::LoadPartitions, request, response, options);
}

Status
MilvusConnection::ReleasePartitions(const proto::milvus::ReleasePartitionsRequest& request,
                                    proto::common::Status& response, const GrpcContextOptions& options) {
    return grpcCall("ReleasePartitions", &Stub::ReleasePartitions, request, response, options);
}

Status
MilvusConnection::GetPartitionStatistics(const proto::milvus::GetPartitionStatisticsRequest& request,
                                         proto::milvus::GetPartitionStatisticsResponse& response,
                                         const GrpcContextOptions& options) {
    return grpcCall("GetPartitionStatistics", &Stub::GetPartitionStatistics, request, response, options);
}

Status
MilvusConnection::CreateAlias(const proto::milvus::CreateAliasRequest& request, proto::common::Status& response,
                              const GrpcContextOptions& options) {
    return grpcCall("CreateAlias", &Stub::CreateAlias, request, response, options);
}

Status
MilvusConnection::DropAlias(const proto::milvus::DropAliasRequest& request, proto::common::Status& response,
                            const GrpcContextOptions& options) {
    return grpcCall("DropAlias", &Stub::DropAlias, request, response, options);
}

Status
MilvusConnection::AlterAlias(const proto::milvus::AlterAliasRequest& request, proto::common::Status& response,
                             const GrpcContextOptions& options) {
    return grpcCall("AlterAlias", &Stub::AlterAlias, request, response, options);
}

Status
MilvusConnection::CreateIndex(const proto::milvus::CreateIndexRequest& request, proto::common::Status& response,
                              const GrpcContextOptions& options) {
    return grpcCall("CreateIndex", &Stub::CreateIndex, request, response, options);
}

Status
MilvusConnection::DescribeIndex(const proto::milvus::DescribeIndexRequest& request,
                                proto::milvus::DescribeIndexResponse& response, const GrpcContextOptions& options) {
    return grpcCall("DescribeIndex", &Stub::DescribeIndex, request, response, options);
}

Status
MilvusConnection::GetIndexState(const proto::milvus::GetIndexStateRequest& request,
                                proto::milvus::GetIndexStateResponse& response, const GrpcContextOptions& options) {
    return grpcCall("GetIndexState", &Stub::GetIndexState, request, response, options);
}

Status
MilvusConnection::GetIndexBuildProgress(const proto::milvus::GetIndexBuildProgressRequest& request,
                                        proto::milvus::GetIndexBuildProgressResponse& response,
                                        const GrpcContextOptions& options) {
    return grpcCall("GetIndexBuildProgress", &Stub::GetIndexBuildProgress, request, response, options);
}

Status
MilvusConnection::DropIndex(const proto::milvus::DropIndexRequest& request, proto::common::Status& response,
                            const GrpcContextOptions& options) {
    return grpcCall("DropIndex", &Stub::DropIndex, request, response, options);
}

Status
MilvusConnection::Flush(const proto::milvus::FlushRequest& request, proto::milvus::FlushResponse& response,
                        const GrpcContextOptions& options) {
    return grpcCall("Flush", &Stub::Flush, request, response, options);
}

Status
MilvusConnection::Insert(const proto::milvus::InsertRequest& request, proto::milvus::MutationResult& response,
                         const GrpcContextOptions& options) {
    return grpcCall("Insert", &Stub::Insert, request, response, options);
}

Status
MilvusConnection::Delete(const proto::milvus::DeleteRequest& request, proto::milvus::MutationResult& response,
                         const GrpcContextOptions& options) {
    return grpcCall("Delete", &Stub::Delete, request, response, options);
}

Status
MilvusConnection::Search(const proto::milvus::SearchRequest& request, proto::milvus::SearchResults& response,
                         const GrpcContextOptions& options) {
    return grpcCall("Search", &Stub::Search, request, response, options);
}

Status
MilvusConnection::Query(const proto::milvus::QueryRequest& request, proto::milvus::QueryResults& response,
                        const GrpcContextOptions& options) {
    return grpcCall("Query", &Stub::Query, request, response, options);
}

Status
MilvusConnection::CalcDistance(const proto::milvus::CalcDistanceRequest& request,
                               proto::milvus::CalcDistanceResults& response, const GrpcContextOptions& options) {
    return grpcCall("CalcDistance", &Stub::CalcDistance, request, response, options);
}

Status
MilvusConnection::GetFlushState(const proto::milvus::GetFlushStateRequest& request,
                                proto::milvus::GetFlushStateResponse& response, const GrpcContextOptions& options) {
    return grpcCall("GetFlushState", &Stub::GetFlushState, request, response, options);
}

Status
MilvusConnection::GetPersistentSegmentInfo(const proto::milvus::GetPersistentSegmentInfoRequest& request,
                                           proto::milvus::GetPersistentSegmentInfoResponse& response,
                                           const GrpcContextOptions& options) {
    return grpcCall("GetPersistentSegmentInfo", &Stub::GetPersistentSegmentInfo, request, response, options);
}

Status
MilvusConnection::GetQuerySegmentInfo(const proto::milvus::GetQuerySegmentInfoRequest& request,
                                      proto::milvus::GetQuerySegmentInfoResponse& response,
                                      const GrpcContextOptions& options) {
    return grpcCall("GetQuerySegmentInfo", &Stub::GetQuerySegmentInfo, request, response, options);
}

Status
MilvusConnection::GetMetrics(const proto::milvus::GetMetricsRequest& request,
                             proto::milvus::GetMetricsResponse& response, const GrpcContextOptions& options) {
    return grpcCall("GetMetrics", &Stub::GetMetrics, request, response, options);
}

Status
MilvusConnection::LoadBalance(const proto::milvus::LoadBalanceRequest& request, proto::common::Status& response,
                              const GrpcContextOptions& options) {
    return grpcCall("LoadBalance", &Stub::LoadBalance, request, response, options);
}

Status
MilvusConnection::GetCompactionState(const proto::milvus::GetCompactionStateRequest& request,
                                     proto::milvus::GetCompactionStateResponse& response,
                                     const GrpcContextOptions& options) {
    return grpcCall("GetCompactionState", &Stub::GetCompactionState, request, response, options);
}

Status
MilvusConnection::ManualCompaction(const proto::milvus::ManualCompactionRequest& request,
                                   proto::milvus::ManualCompactionResponse& response,
                                   const GrpcContextOptions& options) {
    return grpcCall("ManualCompaction", &Stub::ManualCompaction, request, response, options);
}

Status
MilvusConnection::GetCompactionPlans(const proto::milvus::GetCompactionPlansRequest& request,
                                     proto::milvus::GetCompactionPlansResponse& response,
                                     const GrpcContextOptions& options) {
    return grpcCall("GetCompactionPlans", &Stub::GetCompactionStateWithPlans, request, response, options);
}

Status
MilvusConnection::CreateCredential(const proto::milvus::CreateCredentialRequest& request,
                                   proto::common::Status& response, const GrpcContextOptions& options) {
    return grpcCall("CreateCredential", &Stub::CreateCredential, request, response, options);
}

Status
MilvusConnection::UpdateCredential(const proto::milvus::UpdateCredentialRequest& request,
                                   proto::common::Status& response, const GrpcContextOptions& options) {
    return grpcCall("UpdateCredential", &Stub::UpdateCredential, request, response, options);
}

Status
MilvusConnection::DeleteCredential(const proto::milvus::DeleteCredentialRequest& request,
                                   proto::common::Status& response, const GrpcContextOptions& options) {
    return grpcCall("DeleteCredential", &Stub::DeleteCredential, request, response, options);
}

Status
MilvusConnection::ListCredUsers(const proto::milvus::ListCredUsersRequest& request,
                                proto::milvus::ListCredUsersResponse& response, const GrpcContextOptions& options) {
    return grpcCall("ListCredUsers", &Stub::ListCredUsers, request, response, options);
}

}  // namespace milvus
