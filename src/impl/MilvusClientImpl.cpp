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

#include <chrono>
#include <thread>

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
    auto rpc_status = connection_->CreateCollection(rpc_request, response);
    if (!rpc_status.IsOk()) {
        return rpc_status;
    }

    if (response.error_code() != proto::common::ErrorCode::Success) {
        return Status{StatusCode::SERVER_FAILED, response.reason()};
    }

    return rpc_status;
}

Status
MilvusClientImpl::HasCollection(const std::string& collection_name, bool& has) {
    if (connection_ == nullptr) {
        return Status(StatusCode::NOT_CONNECTED, "Connection is not ready!");
    }

    proto::milvus::HasCollectionRequest rpc_request;
    rpc_request.set_collection_name(collection_name);
    rpc_request.set_time_stamp(0);

    proto::milvus::BoolResponse response;
    auto ret = connection_->HasCollection(rpc_request, response);
    has = response.value();
    return ret;
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
    if (connection_ == nullptr) {
        return Status(StatusCode::NOT_CONNECTED, "Connection is not ready!");
    }
    proto::milvus::ShowCollectionsRequest rpc_request;
    if (collection_names.empty()) {
        rpc_request.set_type(proto::milvus::ShowType::All);
    } else {
        rpc_request.set_type(proto::milvus::ShowType::InMemory);
        for (auto& collection_name : collection_names) {
            rpc_request.add_collection_names(collection_name);
        }
    }
    proto::milvus::ShowCollectionsResponse response;
    Status ret = connection_->ShowCollections(rpc_request, response);
    if (ret.IsOk()) {
        for (size_t i = 0; i < response.collection_ids_size(); i++) {
            collections_info.push_back(CollectionInfo(response.collection_names(i), response.collection_ids(i),
                                                      response.created_utc_timestamps(i),
                                                      response.inmemory_percentages(i)));
        }
    }
    return ret;
}

Status
MilvusClientImpl::CreatePartition(const std::string& collection_name, const std::string& partition_name) {
    if (connection_ == nullptr) {
        return Status(StatusCode::NOT_CONNECTED, "Connection is not ready!");
    }

    proto::milvus::CreatePartitionRequest rpc_request;
    rpc_request.set_collection_name(collection_name);
    rpc_request.set_partition_name(partition_name);

    proto::common::Status response;
    return connection_->CreatePartition(rpc_request, response);
}

Status
MilvusClientImpl::DropPartition(const std::string& collection_name, const std::string& partition_name) {
    if (connection_ == nullptr) {
        return Status(StatusCode::NOT_CONNECTED, "Connection is not ready!");
    }

    proto::milvus::DropPartitionRequest rpc_request;
    rpc_request.set_collection_name(collection_name);
    rpc_request.set_partition_name(partition_name);

    proto::common::Status response;
    return connection_->DropPartition(rpc_request, response);
}

Status
MilvusClientImpl::HasPartition(const std::string& collection_name, const std::string& partition_name, bool& has) {
    if (connection_ == nullptr) {
        return Status(StatusCode::NOT_CONNECTED, "Connection is not ready!");
    }

    proto::milvus::HasPartitionRequest rpc_request;
    rpc_request.set_collection_name(collection_name);
    rpc_request.set_partition_name(partition_name);

    proto::milvus::BoolResponse response;
    auto ret = connection_->HasPartition(rpc_request, response);
    has = response.value();
    return ret;
}

Status
MilvusClientImpl::LoadPartitions(const std::string& collection_name, const std::vector<std::string>& partition_names,
                                 const TimeoutSetting& timeout) {
    if (connection_ == nullptr) {
        return Status(StatusCode::NOT_CONNECTED, "Connection is not ready!");
    }

    auto wait_seconds = timeout.WaitingTimeout();
    std::chrono::time_point<std::chrono::steady_clock> started{};
    if (wait_seconds > 0) {
        started = std::chrono::steady_clock::now();
    }

    proto::milvus::LoadPartitionsRequest rpc_request;
    rpc_request.set_collection_name(collection_name);
    for (const auto& partition_name : partition_names) {
        rpc_request.add_partition_names(partition_name);
    }

    proto::common::Status response;
    auto ret = connection_->LoadPartitions(rpc_request, response);
    if (not ret.IsOk() or wait_seconds == 0) {
        return ret;
    }

    waitForStatus(
        [&collection_name, &partition_names, this]() -> Status {
            PartitionsInfo partitions_info;
            auto status = ShowPartitions(collection_name, partition_names, partitions_info);
            if (status.IsOk() and
                std::any_of(partitions_info.begin(), partitions_info.end(),
                            [](const PartitionInfo& partition_info) { return not partition_info.Loaded(); })) {
                return Status{StatusCode::TIMEOUT, "Timeout once"};
            }
            return status;
        },
        started, timeout, ret);
    return ret;
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
    if (connection_ == nullptr) {
        return Status(StatusCode::NOT_CONNECTED, "Connection is not ready!");
    }

    proto::milvus::ShowPartitionsRequest rpc_request;
    rpc_request.set_collection_name(collection_name);
    if (partition_names.empty()) {
        rpc_request.set_type(milvus::proto::milvus::ShowType::All);
    } else {
        rpc_request.set_type(milvus::proto::milvus::ShowType::InMemory);
    }

    for (const auto& partition_name : partition_names) {
        rpc_request.add_partition_names(partition_name);
    }

    proto::milvus::ShowPartitionsResponse response;
    auto ret = connection_->ShowPartitions(rpc_request, response);

    auto count = response.partition_names_size();
    if (count > 0) {
        partitions_info.reserve(count);
    }
    for (size_t i = 0; i < count; ++i) {
        partitions_info.emplace_back(response.partition_names(i), response.partitionids(i),
                                     response.created_timestamps(i), response.inmemory_percentages(i));
    }

    return ret;
}

Status
MilvusClientImpl::CreateAlias(const std::string& collection_name, const std::string& alias) {
    return Status::OK();
}

Status
MilvusClientImpl::DropAlias(const std::string& alias) {
    return Status::OK();
}

Status
MilvusClientImpl::AlterAlias(const std::string& collection_name, const std::string& alias) {
    return Status::OK();
}

Status
MilvusClientImpl::CreateIndex(const std::string& collection_name, const IndexDesc& index_desc) {
    return Status::OK();
}

Status
MilvusClientImpl::DescribeIndex(const std::string& collection_name, const std::string& field_name,
                                IndexDesc& index_desc) {
    return Status::OK();
}

Status
MilvusClientImpl::GetIndexState(const std::string& collection_name, const std::string& field_name, IndexState& state) {
    return Status::OK();
}

Status
MilvusClientImpl::GetIndexBuildProgress(const std::string& collection_name, const std::string& field_name,
                                        IndexProgress& progress) {
    return Status::OK();
}

Status
MilvusClientImpl::DropIndex(const std::string& collection_name, const std::string& field_name) {
    return Status::OK();
}

void
MilvusClientImpl::waitForStatus(std::function<Status()> query_function,
                                const std::chrono::time_point<std::chrono::steady_clock> started,
                                const TimeoutSetting& timeout, Status& status) {
    auto calculated_next_wait = started;
    auto wait_milliseconds = timeout.WaitingTimeout() * 1000;
    auto wait_interval = timeout.WaitingInterval();
    auto final_timeout = started + std::chrono::milliseconds{wait_milliseconds};
    while (wait_milliseconds > 0) {
        calculated_next_wait += std::chrono::milliseconds{wait_interval};
        auto next_wait = std::min(calculated_next_wait, final_timeout);
        std::this_thread::sleep_until(next_wait);

        status = query_function();

        if (status.Code() != StatusCode::TIMEOUT) {
            break;
        }

        if (next_wait == final_timeout) {
            wait_milliseconds = 0;
        } else {
            wait_milliseconds -= wait_interval;
        }
    }
}

}  // namespace milvus
