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

#include "MilvusClientImpl.h"

#include "common.pb.h"
#include "milvus.grpc.pb.h"
#include "milvus.pb.h"
#include "schema.pb.h"

namespace milvus {

std::shared_ptr<MilvusClient>
MilvusClient::Create() {
    return std::make_shared<MilvusClientImpl>();
}

MilvusClientImpl::~MilvusClientImpl() {
    Disconnect();
}

Status
MilvusClientImpl::Connect(const ConnectParam& connect_param) {
    if (connection_ != nullptr) {
        connection_->Disconnect();
    }

    // TODO: check connect parameter

    connection_ = std::make_shared<MilvusConnection>();
    std::string uri = connect_param.host_ + ":" + std::to_string(connect_param.port_);

    return connection_->Connect(uri);
}

Status
MilvusClientImpl::Disconnect() {
    if (connection_ != nullptr) {
        return connection_->Disconnect();
    }

    return Status::OK();
}

Status
MilvusClientImpl::CreateCollection(const CollectionSchema& schema) {
    if (connection_ == nullptr) {
        return Status(StatusCode::NOT_CONNECTED, "Connection is not ready!");
    }

    proto::milvus::CreateCollectionRequest rpc_request;
    rpc_request.set_collection_name(schema.Name());

    proto::schema::CollectionSchema rpc_collection;
    rpc_collection.set_name(schema.Name());
    rpc_collection.set_description(schema.Description());

    for (auto& field : schema.Fields()) {
        proto::schema::FieldSchema* rpc_field = rpc_collection.add_fields();
        rpc_field->set_name(field.Name());
        rpc_field->set_description(field.Description());
        rpc_field->set_data_type(static_cast<proto::schema::DataType>(field.FieldDataType()));
        rpc_field->set_is_primary_key(field.IsPrimaryKey());
        rpc_field->set_autoid(field.AutoID());

        proto::common::KeyValuePair* kv = rpc_field->add_type_params();
        for (auto& pair : field.TypeParams()) {
            kv->set_key(pair.first);
            kv->set_value(pair.second);
        }
    }

    std::string binary;
    rpc_collection.SerializeToString(&binary);
    rpc_request.set_schema(binary);

    proto::common::Status response;
    return connection_->CreateCollection(rpc_request, response);
}

Status
MilvusClientImpl::HasCollection(const std::string& collection_name, bool& has) {
    return Status::OK();
}

Status
MilvusClientImpl::DropCollection(const std::string& collection_name) {
    return Status::OK();
}

Status
MilvusClientImpl::LoadCollection(const std::string& collection_name, const TimeoutSetting* timeout) {
    return Status::OK();
}

Status
MilvusClientImpl::ReleaseCollection(const std::string& collection_name) {
    return Status::OK();
}

Status
MilvusClientImpl::DescribeCollection(const std::string& collection_name, CollectionDesc& collection_desc) {
    return Status::OK();
}

Status
MilvusClientImpl::GetCollectionStatistics(const std::string& collection_name, const TimeoutSetting* timeout,
                                          CollectionStat& collection_stat) {
    return Status::OK();
}

Status
MilvusClientImpl::ShowCollections(const std::vector<std::string>& collection_names, CollectionsInfo& collections_info) {
    return Status::OK();
}

Status
MilvusClientImpl::CreatePartition(const std::string& collection_name, const std::string& partition_name) {
    return Status::OK();
}

Status
MilvusClientImpl::DropPartition(const std::string& collection_name, const std::string& partition_name) {
    return Status::OK();
}

Status
MilvusClientImpl::HasPartition(const std::string& collection_name, const std::string& partition_name, bool& has) {
    return Status::OK();
}

Status
MilvusClientImpl::LoadPartitions(const std::string& collection_name, const std::vector<std::string>& partition_names,
                                 const TimeoutSetting* timeout) {
    return Status::OK();
}

Status
MilvusClientImpl::ReleasePartitions(const std::string& collection_name,
                                    const std::vector<std::string>& partition_names) {
    return Status::OK();
}

Status
MilvusClientImpl::GetPartitionStatistics(const std::string& collection_name, const std::string& partition_name,
                                         const TimeoutSetting* timeout, PartitionStat& partition_stat) {
    return Status::OK();
}

Status
MilvusClientImpl::ShowPartitions(const std::string& collection_name, const std::vector<std::string>& partition_names,
                                 PartitionsInfo& partitions_info) {
    return Status::OK();
}

}  // namespace milvus
