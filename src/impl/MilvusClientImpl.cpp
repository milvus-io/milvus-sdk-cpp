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

// This macro is to check two kinds of status:
//  1. Status returned by lower methods
//  2. Proto status returned by server side
#define ON_ERROR_RETURN(status, resp_status)                                 \
    {                                                                        \
        if (!status.IsOk()) {                                                \
            return status;                                                   \
        }                                                                    \
        if (resp_status.error_code() != proto::common::ErrorCode::Success) { \
            return Status{StatusCode::SERVER_FAILED, resp_status.reason()};  \
        }                                                                    \
    }

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
        return {StatusCode::NOT_CONNECTED, "Connection is not ready!"};
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
    auto ret = connection_->CreateCollection(rpc_request, response);
    ON_ERROR_RETURN(ret, response);

    return ret;
}

Status
MilvusClientImpl::HasCollection(const std::string& collection_name, bool& has) {
    if (connection_ == nullptr) {
        return {StatusCode::NOT_CONNECTED, "Connection is not ready!"};
    }

    proto::milvus::HasCollectionRequest rpc_request;
    rpc_request.set_collection_name(collection_name);
    rpc_request.set_time_stamp(0);

    proto::milvus::BoolResponse response;
    auto ret = connection_->HasCollection(rpc_request, response);
    ON_ERROR_RETURN(ret, response.status());
    has = response.value();

    return ret;
}

Status
MilvusClientImpl::DropCollection(const std::string& collection_name) {
    return Status::OK();
}

Status
MilvusClientImpl::LoadCollection(const std::string& collection_name, const ProgressMonitor& progress_monitor) {
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
MilvusClientImpl::GetCollectionStatistics(const std::string& collection_name, const ProgressMonitor& progress_monitor,
                                          CollectionStat& collection_stat) {
    return Status::OK();
}

Status
MilvusClientImpl::ShowCollections(const std::vector<std::string>& collection_names, CollectionsInfo& collections_info) {
    if (connection_ == nullptr) {
        return {StatusCode::NOT_CONNECTED, "Connection is not ready!"};
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
    ON_ERROR_RETURN(ret, response.status());

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
        return {StatusCode::NOT_CONNECTED, "Connection is not ready!"};
    }

    proto::milvus::CreatePartitionRequest rpc_request;
    rpc_request.set_collection_name(collection_name);
    rpc_request.set_partition_name(partition_name);

    proto::common::Status response;
    auto ret = connection_->CreatePartition(rpc_request, response);
    ON_ERROR_RETURN(ret, response);

    return ret;
}

Status
MilvusClientImpl::DropPartition(const std::string& collection_name, const std::string& partition_name) {
    if (connection_ == nullptr) {
        return {StatusCode::NOT_CONNECTED, "Connection is not ready!"};
    }

    proto::milvus::DropPartitionRequest rpc_request;
    rpc_request.set_collection_name(collection_name);
    rpc_request.set_partition_name(partition_name);

    proto::common::Status response;
    auto ret = connection_->DropPartition(rpc_request, response);
    ON_ERROR_RETURN(ret, response);

    return ret;
}

Status
MilvusClientImpl::HasPartition(const std::string& collection_name, const std::string& partition_name, bool& has) {
    if (connection_ == nullptr) {
        return {StatusCode::NOT_CONNECTED, "Connection is not ready!"};
    }

    proto::milvus::HasPartitionRequest rpc_request;
    rpc_request.set_collection_name(collection_name);
    rpc_request.set_partition_name(partition_name);

    proto::milvus::BoolResponse response;
    auto ret = connection_->HasPartition(rpc_request, response);
    ON_ERROR_RETURN(ret, response.status());
    has = response.value();

    return ret;
}

Status
MilvusClientImpl::LoadPartitions(const std::string& collection_name, const std::vector<std::string>& partition_names,
                                 const ProgressMonitor& progress_monitor) {
    if (connection_ == nullptr) {
        return {StatusCode::NOT_CONNECTED, "Connection is not ready!"};
    }

    auto wait_seconds = progress_monitor.CheckTimeout();
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
    ON_ERROR_RETURN(ret, response);

    if (wait_seconds == 0) {
        return ret;
    }

    waitForStatus(
        [&collection_name, &partition_names, this](Progress& progress) -> Status {
            PartitionsInfo partitions_info;
            auto status = ShowPartitions(collection_name, partition_names, partitions_info);
            if (not status.IsOk()) {
                return status;
            }
            progress.total_ = partition_names.size();
            progress.finished_ =
                std::count_if(partitions_info.begin(), partitions_info.end(),
                              [](const PartitionInfo& partition_info) { return partition_info.Loaded(); });
            if (progress.total_ != progress.finished_) {
                return Status{StatusCode::TIMEOUT, "not all partitions finished"};
            }
            return status;
        },
        started, progress_monitor, ret);
    return ret;
}

Status
MilvusClientImpl::ReleasePartitions(const std::string& collection_name,
                                    const std::vector<std::string>& partition_names) {
    if (connection_ == nullptr) {
        return {StatusCode::NOT_CONNECTED, "Connection is not ready!"};
    }

    proto::milvus::ReleasePartitionsRequest rpc_request;
    rpc_request.set_collection_name(collection_name);
    for (const auto& partition_name : partition_names) {
        rpc_request.add_partition_names(partition_name);
    }

    proto::common::Status response;
    auto ret = connection_->ReleasePartitions(rpc_request, response);
    ON_ERROR_RETURN(ret, response);

    return ret;
}

Status
MilvusClientImpl::GetPartitionStatistics(const std::string& collection_name, const std::string& partition_name,
                                         const ProgressMonitor& progress_monitor, PartitionStat& partition_stat) {
    return Status::OK();
}

Status
MilvusClientImpl::ShowPartitions(const std::string& collection_name, const std::vector<std::string>& partition_names,
                                 PartitionsInfo& partitions_info) {
    if (connection_ == nullptr) {
        return {StatusCode::NOT_CONNECTED, "Connection is not ready!"};
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
    ON_ERROR_RETURN(ret, response.status());

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
MilvusClientImpl::CreateIndex(const std::string& collection_name, const IndexDesc& index_desc,
                              const ProgressMonitor& progress_monitor) {
    if (connection_ == nullptr) {
        return {StatusCode::NOT_CONNECTED, "Connection is not ready!"};
    }
    proto::milvus::CreateIndexRequest rpc_request;
    rpc_request.set_collection_name(collection_name);
    rpc_request.set_field_name(index_desc.GetFieldName());

    proto::common::Status response;
    auto ret = connection_->CreateIndex(rpc_request, response);
    ON_ERROR_RETURN(ret, response);

    if (progress_monitor.CheckTimeout() == 0) {
        return ret;
    }
    waitForStatus(
        [&collection_name, &index_desc, ret, this](Progress& progress) -> Status {
            IndexDesc index_info;
            auto status = DescribeIndex(collection_name, index_desc.GetFieldName(), index_info);
            if (not status.IsOk()) {
                return status;
            }
            progress.total_ = 1;
            progress.finished_ = ret.IsOk() ? 1 : 0;
            if (progress.total_ != progress.finished_) {
                return Status{StatusCode::TIMEOUT, "not all indexes finished"};
            }
            return status;
        },
        std::chrono::steady_clock::now(), progress_monitor, ret);
    return ret;
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

Status
MilvusClientImpl::Insert(const std::string& collection_name, const std::string& partition_name,
                         const std::vector<FieldDataPtr>& fields, IDArray& id_array) {
    if (connection_ == nullptr) {
        return {StatusCode::NOT_CONNECTED, "Connection is not ready!"};
    }

    proto::milvus::InsertRequest rpc_request;
    // TODO(matrixji): add common validations check for fields
    // TODO(matrixji): add scheme based validations check for fields

    auto* mutable_fields = rpc_request.mutable_fields_data();
    rpc_request.set_collection_name(collection_name);
    rpc_request.set_partition_name(partition_name);
    rpc_request.set_num_rows((**fields.begin()).Count());
    for (const auto& field : fields) {
        mutable_fields->Add(std::move(fieldCast(*field)));
    }

    milvus::proto::milvus::MutationResult response;
    auto ret = connection_->Insert(rpc_request, response);
    ON_ERROR_RETURN(ret, response.status());

    const auto& ids = response.ids();
    // milvus 2.0: only int ids so far
    if (ids.has_int_id()) {
        std::vector<int64_t> int_array;
        auto& int_ids = ids.int_id();
        int_array.reserve(int_ids.data_size());
        std::copy(int_ids.data().begin(), int_ids.data().end(), std::back_inserter(int_array));
        id_array = IDArray(int_array);
    }

    return ret;
}

Status
MilvusClientImpl::flush(const std::vector<std::string>& collection_names, const ProgressMonitor& progress_monitor) {
    if (connection_ == nullptr) {
        return {StatusCode::NOT_CONNECTED, "Connection is not ready!"};
    }

    proto::milvus::FlushRequest rpc_request;
    for (const auto& collection_name : collection_names) {
        rpc_request.add_collection_names(collection_name);
    }

    auto wait_seconds = progress_monitor.CheckTimeout();
    std::chrono::time_point<std::chrono::steady_clock> started{};
    if (wait_seconds > 0) {
        started = std::chrono::steady_clock::now();
    }

    proto::milvus::FlushResponse response;
    auto ret = connection_->Flush(rpc_request, response);
    ON_ERROR_RETURN(ret, response.status());

    waitForStatus(
        [&](Progress&) -> Status {
            Status status;

            // TODO: call GetPersistentSegmentInfo() to check segment state

            return status;
        },
        started, progress_monitor, ret);

    return Status::OK();
}

void
MilvusClientImpl::waitForStatus(std::function<Status(Progress&)> query_function,
                                const std::chrono::time_point<std::chrono::steady_clock> started,
                                const ProgressMonitor& progress_monitor, Status& status) {
    auto calculated_next_wait = started;
    auto wait_milliseconds = progress_monitor.CheckTimeout() * 1000;
    auto wait_interval = progress_monitor.CheckInterval();
    auto final_timeout = started + std::chrono::milliseconds{wait_milliseconds};
    while (wait_milliseconds > 0) {
        calculated_next_wait += std::chrono::milliseconds{wait_interval};
        auto next_wait = std::min(calculated_next_wait, final_timeout);
        std::this_thread::sleep_until(next_wait);

        Progress current_progress;
        status = query_function(current_progress);
        if (status.Code() != StatusCode::TIMEOUT) {
            break;
        }

        progress_monitor.DoProgress(current_progress);

        if (next_wait == final_timeout) {
            wait_milliseconds = 0;
        } else {
            wait_milliseconds -= wait_interval;
        }
    }
}

milvus::proto::schema::DataType
dataTypeCast(DataType type) {
    switch (type) {
        case DataType::BOOL:
            return milvus::proto::schema::DataType::Bool;
            break;
        case DataType::INT8:
            return milvus::proto::schema::DataType::Int8;
            break;
        case DataType::INT16:
            return milvus::proto::schema::DataType::Int16;
            break;
        case DataType::INT32:
            return milvus::proto::schema::DataType::Int32;
            break;
        case DataType::INT64:
            return milvus::proto::schema::DataType::Int64;
            break;
        case DataType::FLOAT:
            return milvus::proto::schema::DataType::Float;
            break;
        case DataType::DOUBLE:
            return milvus::proto::schema::DataType::Double;
            break;
        case DataType::STRING:
            return milvus::proto::schema::DataType::String;
            break;
        case DataType::BINARY_VECTOR:
            return milvus::proto::schema::DataType::BinaryVector;
            break;
        case DataType::FLOAT_VECTOR:
            return milvus::proto::schema::DataType::FloatVector;
            break;
        default:
            return milvus::proto::schema::DataType::None;
    }
}

void
fillScalarsData(const BoolFieldData& field, milvus::proto::schema::ScalarField& scalars) {
    auto& data = field.Data();
    auto& scalars_data = *(scalars.mutable_bool_data()->mutable_data());
    scalars_data.Add(data.begin(), data.end());
}

void
fillScalarsData(const Int8FieldData& field, milvus::proto::schema::ScalarField& scalars) {
    auto& data = field.Data();
    auto& scalars_data = *(scalars.mutable_int_data()->mutable_data());
    scalars_data.Add(data.begin(), data.end());
}

void
fillScalarsData(const Int16FieldData& field, milvus::proto::schema::ScalarField& scalars) {
    auto& data = field.Data();
    auto& scalars_data = *(scalars.mutable_int_data()->mutable_data());
    scalars_data.Add(data.begin(), data.end());
}

void
fillScalarsData(const Int32FieldData& field, milvus::proto::schema::ScalarField& scalars) {
    auto& data = field.Data();
    auto& scalars_data = *(scalars.mutable_int_data()->mutable_data());
    scalars_data.Add(data.begin(), data.end());
}

void
fillScalarsData(const Int64FieldData& field, milvus::proto::schema::ScalarField& scalars) {
    auto& data = field.Data();
    auto& scalars_data = *(scalars.mutable_long_data()->mutable_data());
    scalars_data.Add(data.begin(), data.end());
}

void
fillScalarsData(const FloatFieldData& field, milvus::proto::schema::ScalarField& scalars) {
    auto& data = field.Data();
    auto& scalars_data = *(scalars.mutable_float_data()->mutable_data());
    scalars_data.Add(data.begin(), data.end());
}
void
fillScalarsData(const DoubleFieldData& field, milvus::proto::schema::ScalarField& scalars) {
    auto& data = field.Data();
    auto& scalars_data = *(scalars.mutable_double_data()->mutable_data());
    scalars_data.Add(data.begin(), data.end());
}

void
fillScalarsData(const StringFieldData& field, milvus::proto::schema::ScalarField& scalars) {
    auto& data = field.Data();
    auto& scalars_data = *(scalars.mutable_string_data()->mutable_data());
    scalars_data.Reserve(data.size());
    for (auto item : data) {
        scalars_data.Add(std::move(item));
    }
}

void
fillVectorsData(const BinaryVecFieldData& field, milvus::proto::schema::VectorField& vectors) {
    auto& vectors_data = *(vectors.mutable_binary_vector());
    auto& data = field.Data();
    size_t dim = data.begin()->size();
    vectors_data.reserve(data.size() * dim);
    for (const auto& item : data) {
        std::copy(item.begin(), item.end(), std::back_inserter(vectors_data));
    }
    vectors.set_dim(dim);
}

void
fillVectorsData(const FloatVecFieldData& field, milvus::proto::schema::VectorField& vectors) {
    auto& vectors_data = *(vectors.mutable_float_vector()->mutable_data());
    auto& data = field.Data();
    size_t dim = data.begin()->size();
    vectors_data.Reserve(data.size() * dim);
    size_t copy_index = 0;
    for (const auto& item : data) {
        vectors_data.Add(item.begin(), item.end());
    }
    vectors.set_dim(dim);
}

milvus::proto::schema::FieldData
MilvusClientImpl::fieldCast(const Field& field) {
    milvus::proto::schema::FieldData field_data;
    auto field_type = field.Type();
    field_data.set_field_name(field.Name());
    field_data.set_type(dataTypeCast(field_type));

    switch (field_type) {
        case DataType::BINARY_VECTOR: {
            auto& vectors = *field_data.mutable_vectors();
            const auto* original_field = dynamic_cast<const BinaryVecFieldData*>(&field);
            if (original_field) {
                fillVectorsData(*original_field, vectors);
            }
        } break;

        case DataType::FLOAT_VECTOR: {
            auto& vectors = *field_data.mutable_vectors();
            const auto* original_field = dynamic_cast<const FloatVecFieldData*>(&field);
            if (original_field) {
                fillVectorsData(*original_field, vectors);
            }
        } break;

        case DataType::BOOL: {
            auto& scalars = *field_data.mutable_scalars();
            const auto* original_field = dynamic_cast<const BoolFieldData*>(&field);
            if (original_field) {
                fillScalarsData(*original_field, scalars);
            }
        } break;

        case DataType::INT8: {
            auto& scalars = *field_data.mutable_scalars();
            const auto* original_field = dynamic_cast<const Int8FieldData*>(&field);
            if (original_field) {
                fillScalarsData(*original_field, scalars);
            }
        } break;
        case DataType::INT16: {
            auto& scalars = *field_data.mutable_scalars();
            const auto* original_field = dynamic_cast<const Int16FieldData*>(&field);
            if (original_field) {
                fillScalarsData(*original_field, scalars);
            }
        } break;
        case DataType::INT32: {
            auto& scalars = *field_data.mutable_scalars();
            const auto* original_field = dynamic_cast<const Int32FieldData*>(&field);
            if (original_field) {
                fillScalarsData(*original_field, scalars);
            }
        } break;
        case DataType::INT64: {
            auto& scalars = *field_data.mutable_scalars();
            const auto* original_field = dynamic_cast<const Int64FieldData*>(&field);
            if (original_field) {
                fillScalarsData(*original_field, scalars);
            }
        } break;
        case DataType::FLOAT: {
            auto& scalars = *field_data.mutable_scalars();
            const auto* original_field = dynamic_cast<const FloatFieldData*>(&field);
            if (original_field) {
                fillScalarsData(*original_field, scalars);
            }
        } break;
        case DataType::DOUBLE: {
            auto& scalars = *field_data.mutable_scalars();
            const auto* original_field = dynamic_cast<const DoubleFieldData*>(&field);
            if (original_field) {
                fillScalarsData(*original_field, scalars);
            }
        } break;
        case DataType::STRING: {
            auto& scalars = *field_data.mutable_scalars();
            const auto* original_field = dynamic_cast<const StringFieldData*>(&field);
            if (original_field) {
                fillScalarsData(*original_field, scalars);
            }
        } break;

        default:
            break;
    }

    return field_data;
}

}  // namespace milvus
