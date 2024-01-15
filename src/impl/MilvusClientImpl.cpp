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
#include <nlohmann/json.hpp>
#include <thread>

#include "TypeUtils.h"
#include "common.pb.h"
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
MilvusClientImpl::Connect(const ConnectParam& param) {
    if (connection_ != nullptr) {
        connection_->Disconnect();
    }

    // TODO: check connect parameter
    connection_ = std::make_shared<MilvusConnection>();
    return connection_->Connect(param);
}

Status
MilvusClientImpl::Disconnect() {
    if (connection_ != nullptr) {
        return connection_->Disconnect();
    }

    return Status::OK();
}

Status
MilvusClientImpl::GetVersion(std::string& version) {
    auto pre = []() {
        proto::milvus::GetVersionRequest rpc_request;
        return rpc_request;
    };

    auto post = [&version](const proto::milvus::GetVersionResponse& response) { version = response.version(); };

    return apiHandler<proto::milvus::GetVersionRequest, proto::milvus::GetVersionResponse>(
        pre, &MilvusConnection::GetVersion, post);
}

Status
MilvusClientImpl::CreateCollection(const CollectionSchema& schema) {
    auto pre = [&schema]() {
        proto::milvus::CreateCollectionRequest rpc_request;
        rpc_request.set_collection_name(schema.Name());
        rpc_request.set_shards_num(schema.ShardsNum());
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
        return rpc_request;
    };

    return apiHandler<proto::milvus::CreateCollectionRequest, proto::common::Status>(
        pre, &MilvusConnection::CreateCollection);
}

Status
MilvusClientImpl::HasCollection(const std::string& collection_name, bool& has) {
    auto pre = [&collection_name]() {
        proto::milvus::HasCollectionRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_time_stamp(0);
        return rpc_request;
    };

    auto post = [&has](const proto::milvus::BoolResponse& response) { has = response.value(); };

    return apiHandler<proto::milvus::HasCollectionRequest, proto::milvus::BoolResponse>(
        pre, &MilvusConnection::HasCollection, post);
}

Status
MilvusClientImpl::DropCollection(const std::string& collection_name) {
    auto pre = [&collection_name]() {
        proto::milvus::DropCollectionRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::DropCollectionRequest, proto::common::Status>(pre,
                                                                                   &MilvusConnection::DropCollection);
}

Status
MilvusClientImpl::LoadCollection(const std::string& collection_name, int replica_number,
                                 const ProgressMonitor& progress_monitor) {
    auto pre = [&collection_name, replica_number]() {
        proto::milvus::LoadCollectionRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_replica_number(replica_number);
        return rpc_request;
    };

    auto wait_for_status = [this, &collection_name, &progress_monitor](const proto::common::Status&) {
        return WaitForStatus(
            [&collection_name, this](Progress& progress) -> Status {
                CollectionsInfo collections_info;
                auto collection_names = std::vector<std::string>{collection_name};
                auto status = ShowCollections(collection_names, collections_info);
                if (!status.IsOk()) {
                    return status;
                }
                progress.total_ = collections_info.size();
                progress.finished_ = std::count_if(
                    collections_info.begin(), collections_info.end(),
                    [](const CollectionInfo& collection_info) { return collection_info.MemoryPercentage() >= 100; });
                return status;
            },
            progress_monitor);
    };

    return apiHandler<proto::milvus::LoadCollectionRequest, proto::common::Status>(
        pre, &MilvusConnection::LoadCollection, wait_for_status);
}

Status
MilvusClientImpl::ReleaseCollection(const std::string& collection_name) {
    auto pre = [&collection_name]() {
        proto::milvus::ReleaseCollectionRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::ReleaseCollectionRequest, proto::common::Status>(
        pre, &MilvusConnection::ReleaseCollection);
}

Status
MilvusClientImpl::DescribeCollection(const std::string& collection_name, CollectionDesc& collection_desc) {
    auto pre = [&collection_name]() {
        proto::milvus::DescribeCollectionRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        return rpc_request;
    };

    auto post = [&collection_desc](const proto::milvus::DescribeCollectionResponse& response) {
        CollectionSchema schema;
        ConvertCollectionSchema(response.schema(), schema);
        schema.SetShardsNum(response.shards_num());
        collection_desc.SetSchema(std::move(schema));
        collection_desc.SetID(response.collectionid());

        std::vector<std::string> aliases;
        aliases.reserve(response.aliases_size());
        aliases.insert(aliases.end(), response.aliases().begin(), response.aliases().end());

        collection_desc.SetAlias(std::move(aliases));
        collection_desc.SetCreatedTime(response.created_timestamp());
    };

    return apiHandler<proto::milvus::DescribeCollectionRequest, proto::milvus::DescribeCollectionResponse>(
        pre, &MilvusConnection::DescribeCollection, post);
}

Status
MilvusClientImpl::RenameCollection(const std::string& collection_name, const std::string& new_collection_name) {
    auto pre = [&collection_name, &new_collection_name]() {
        proto::milvus::RenameCollectionRequest rpc_request;
        rpc_request.set_oldname(collection_name);
        rpc_request.set_newname(new_collection_name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::RenameCollectionRequest, proto::common::Status>(
        pre, &MilvusConnection::RenameCollection);
}

Status
MilvusClientImpl::GetCollectionStatistics(const std::string& collection_name, CollectionStat& collection_stat,
                                          const ProgressMonitor& progress_monitor) {
    auto validate = [&collection_name, &progress_monitor, this]() {
        Status ret;
        if (progress_monitor.CheckTimeout() > 0) {
            ret = Flush(std::vector<std::string>{collection_name}, progress_monitor);
        }
        return ret;
    };

    auto pre = [&collection_name]() {
        proto::milvus::GetCollectionStatisticsRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        return rpc_request;
    };

    auto post = [&collection_stat, &collection_name](const proto::milvus::GetCollectionStatisticsResponse& response) {
        collection_stat.SetName(collection_name);
        for (const auto& stat_pair : response.stats()) {
            collection_stat.Emplace(stat_pair.key(), stat_pair.value());
        }
    };

    return apiHandler<proto::milvus::GetCollectionStatisticsRequest, proto::milvus::GetCollectionStatisticsResponse>(
        validate, pre, &MilvusConnection::GetCollectionStatistics, nullptr, post);
}

Status
MilvusClientImpl::ShowCollections(const std::vector<std::string>& collection_names, CollectionsInfo& collections_info) {
    auto pre = [&collection_names]() {
        proto::milvus::ShowCollectionsRequest rpc_request;

        if (collection_names.empty()) {
            rpc_request.set_type(proto::milvus::ShowType::All);
        } else {
            rpc_request.set_type(proto::milvus::ShowType::InMemory);
            for (auto& collection_name : collection_names) {
                rpc_request.add_collection_names(collection_name);
            }
        }
        return rpc_request;
    };

    auto post = [&collections_info](const proto::milvus::ShowCollectionsResponse& response) {
        for (int i = 0; i < response.collection_ids_size(); i++) {
            auto inmemory_percentage = 0;
            if (response.inmemory_percentages_size() > i) {
                inmemory_percentage = response.inmemory_percentages(i);
            }
            collections_info.emplace_back(response.collection_names(i), response.collection_ids(i),
                                          response.created_utc_timestamps(i), inmemory_percentage);
        }
    };
    return apiHandler<proto::milvus::ShowCollectionsRequest, proto::milvus::ShowCollectionsResponse>(
        pre, &MilvusConnection::ShowCollections, post);
}

Status
MilvusClientImpl::CreatePartition(const std::string& collection_name, const std::string& partition_name) {
    auto pre = [&collection_name, &partition_name]() {
        proto::milvus::CreatePartitionRequest rpc_request;

        rpc_request.set_collection_name(collection_name);
        rpc_request.set_partition_name(partition_name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::CreatePartitionRequest, proto::common::Status>(pre,
                                                                                    &MilvusConnection::CreatePartition);
}

Status
MilvusClientImpl::DropPartition(const std::string& collection_name, const std::string& partition_name) {
    auto pre = [&collection_name, &partition_name]() {
        proto::milvus::DropPartitionRequest rpc_request;

        rpc_request.set_collection_name(collection_name);
        rpc_request.set_partition_name(partition_name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::DropPartitionRequest, proto::common::Status>(pre,
                                                                                  &MilvusConnection::DropPartition);
}

Status
MilvusClientImpl::HasPartition(const std::string& collection_name, const std::string& partition_name, bool& has) {
    auto pre = [&collection_name, &partition_name]() {
        proto::milvus::HasPartitionRequest rpc_request;

        rpc_request.set_collection_name(collection_name);
        rpc_request.set_partition_name(partition_name);
        return rpc_request;
    };

    auto post = [&has](const proto::milvus::BoolResponse& response) { has = response.value(); };

    return apiHandler<proto::milvus::HasPartitionRequest, proto::milvus::BoolResponse>(
        pre, &MilvusConnection::HasPartition, post);
}

Status
MilvusClientImpl::LoadPartitions(const std::string& collection_name, const std::vector<std::string>& partition_names,
                                 int replica_number, const ProgressMonitor& progress_monitor) {
    auto pre = [&collection_name, &partition_names, replica_number]() {
        proto::milvus::LoadPartitionsRequest rpc_request;

        rpc_request.set_collection_name(collection_name);
        for (const auto& partition_name : partition_names) {
            rpc_request.add_partition_names(partition_name);
        }
        rpc_request.set_replica_number(replica_number);
        return rpc_request;
    };

    auto wait_for_status = [this, &collection_name, &partition_names, &progress_monitor](const proto::common::Status&) {
        return WaitForStatus(
            [&collection_name, &partition_names, this](Progress& progress) -> Status {
                PartitionsInfo partitions_info;
                auto status = ShowPartitions(collection_name, partition_names, partitions_info);
                if (!status.IsOk()) {
                    return status;
                }
                progress.total_ = partition_names.size();
                progress.finished_ =
                    std::count_if(partitions_info.begin(), partitions_info.end(),
                                  [](const PartitionInfo& partition_info) { return partition_info.Loaded(); });

                return status;
            },
            progress_monitor);
    };
    return apiHandler<proto::milvus::LoadPartitionsRequest, proto::common::Status>(
        nullptr, pre, &MilvusConnection::LoadPartitions, wait_for_status, nullptr);
}

Status
MilvusClientImpl::ReleasePartitions(const std::string& collection_name,
                                    const std::vector<std::string>& partition_names) {
    auto pre = [&collection_name, &partition_names]() {
        proto::milvus::ReleasePartitionsRequest rpc_request;

        rpc_request.set_collection_name(collection_name);
        for (const auto& partition_name : partition_names) {
            rpc_request.add_partition_names(partition_name);
        }
        return rpc_request;
    };

    return apiHandler<proto::milvus::ReleasePartitionsRequest, proto::common::Status>(
        pre, &MilvusConnection::ReleasePartitions);
}

Status
MilvusClientImpl::GetPartitionStatistics(const std::string& collection_name, const std::string& partition_name,
                                         PartitionStat& partition_stat, const ProgressMonitor& progress_monitor) {
    // do flush in validate stage if needed
    auto validate = [&collection_name, &progress_monitor, this] {
        Status ret;
        if (progress_monitor.CheckTimeout() > 0) {
            ret = Flush(std::vector<std::string>{collection_name}, progress_monitor);
        }
        return ret;
    };

    auto pre = [&collection_name, &partition_name] {
        proto::milvus::GetPartitionStatisticsRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_partition_name(partition_name);
        return rpc_request;
    };

    auto post = [&partition_stat, &partition_name](const proto::milvus::GetPartitionStatisticsResponse& response) {
        partition_stat.SetName(partition_name);
        for (const auto& stat_pair : response.stats()) {
            partition_stat.Emplace(stat_pair.key(), stat_pair.value());
        }
    };

    return apiHandler<proto::milvus::GetPartitionStatisticsRequest, proto::milvus::GetPartitionStatisticsResponse>(
        validate, pre, &MilvusConnection::GetPartitionStatistics, nullptr, post);
}

Status
MilvusClientImpl::ShowPartitions(const std::string& collection_name, const std::vector<std::string>& partition_names,
                                 PartitionsInfo& partitions_info) {
    auto pre = [&collection_name, &partition_names] {
        proto::milvus::ShowPartitionsRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        if (partition_names.empty()) {
            rpc_request.set_type(proto::milvus::ShowType::All);
        } else {
            rpc_request.set_type(proto::milvus::ShowType::InMemory);
        }

        for (const auto& partition_name : partition_names) {
            rpc_request.add_partition_names(partition_name);
        }

        return rpc_request;
    };

    auto post = [&partitions_info](const proto::milvus::ShowPartitionsResponse& response) {
        auto count = response.partition_names_size();
        if (count > 0) {
            partitions_info.reserve(count);
        }
        for (int i = 0; i < count; ++i) {
            int inmemory_percentage = 0;
            if (response.inmemory_percentages_size() > i) {
                inmemory_percentage = response.inmemory_percentages(i);
            }
            partitions_info.emplace_back(response.partition_names(i), response.partitionids(i),
                                         response.created_timestamps(i), inmemory_percentage);
        }
    };

    return apiHandler<proto::milvus::ShowPartitionsRequest, proto::milvus::ShowPartitionsResponse>(
        pre, &MilvusConnection::ShowPartitions, post);
}

Status
MilvusClientImpl::CreateAlias(const std::string& collection_name, const std::string& alias) {
    auto pre = [&collection_name, &alias]() {
        proto::milvus::CreateAliasRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_alias(alias);
        return rpc_request;
    };

    return apiHandler<proto::milvus::CreateAliasRequest, proto::common::Status>(pre, &MilvusConnection::CreateAlias);
}

Status
MilvusClientImpl::DropAlias(const std::string& alias) {
    auto pre = [&alias]() {
        proto::milvus::DropAliasRequest rpc_request;
        rpc_request.set_alias(alias);
        return rpc_request;
    };

    return apiHandler<proto::milvus::DropAliasRequest, proto::common::Status>(pre, &MilvusConnection::DropAlias);
}

Status
MilvusClientImpl::AlterAlias(const std::string& collection_name, const std::string& alias) {
    auto pre = [&collection_name, &alias]() {
        proto::milvus::AlterAliasRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_alias(alias);
        return rpc_request;
    };

    return apiHandler<proto::milvus::AlterAliasRequest, proto::common::Status>(pre, &MilvusConnection::AlterAlias);
}

Status
MilvusClientImpl::CreateIndex(const std::string& collection_name, const IndexDesc& index_desc,
                              const ProgressMonitor& progress_monitor) {
    auto validate = [&index_desc] { return index_desc.Validate(); };

    // flush before create index
    auto flush_status = Flush(std::vector<std::string>{collection_name}, progress_monitor);
    if (!flush_status.IsOk()) {
        return flush_status;
    }

    auto pre = [&collection_name, &index_desc]() {
        proto::milvus::CreateIndexRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_field_name(index_desc.FieldName());

        auto kv_pair = rpc_request.add_extra_params();
        kv_pair->set_key(milvus::KeyIndexType());
        kv_pair->set_value(std::to_string(index_desc.IndexType()));

        kv_pair = rpc_request.add_extra_params();
        kv_pair->set_key(milvus::KeyMetricType());
        kv_pair->set_value(std::to_string(index_desc.MetricType()));

        kv_pair = rpc_request.add_extra_params();
        kv_pair->set_key(milvus::KeyParams());
        kv_pair->set_value(index_desc.ExtraParams());

        return rpc_request;
    };

    auto wait_for_status = [&collection_name, &index_desc, &progress_monitor, this](const proto::common::Status&) {
        return WaitForStatus(
            [&collection_name, &index_desc, this](Progress& progress) -> Status {
                IndexState index_state;
                auto status = GetIndexState(collection_name, index_desc.FieldName(), index_state);
                if (!status.IsOk()) {
                    return status;
                }

                progress.total_ = 100;

                // if index finished, progress set to 100%
                // else if index failed, return error status
                // else if index is in progressing, continue to check
                if (index_state.StateCode() == IndexStateCode::FINISHED ||
                    index_state.StateCode() == IndexStateCode::NONE) {
                    progress.finished_ = 100;
                } else if (index_state.StateCode() == IndexStateCode::FAILED) {
                    return Status{StatusCode::SERVER_FAILED, "index failed:" + index_state.FailedReason()};
                }

                return status;
            },
            progress_monitor);
    };
    return apiHandler<proto::milvus::CreateIndexRequest, proto::common::Status>(
        validate, pre, &MilvusConnection::CreateIndex, wait_for_status, nullptr);
}

Status
MilvusClientImpl::DescribeIndex(const std::string& collection_name, const std::string& field_name,
                                IndexDesc& index_desc) {
    auto pre = [&collection_name, &field_name]() {
        proto::milvus::DescribeIndexRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_field_name(field_name);
        return rpc_request;
    };

    auto post = [&index_desc](const proto::milvus::DescribeIndexResponse& response) {
        auto count = response.index_descriptions_size();
        for (int i = 0; i < count; ++i) {
            auto& field_name = response.index_descriptions(i).field_name();
            auto& index_name = response.index_descriptions(i).index_name();
            index_desc.SetFieldName(field_name);
            index_desc.SetIndexName(index_name);
            auto index_params_size = response.index_descriptions(i).params_size();
            for (int j = 0; j < index_params_size; ++j) {
                const auto& key = response.index_descriptions(i).params(j).key();
                const auto& value = response.index_descriptions(i).params(j).value();
                if (key == milvus::KeyIndexType()) {
                    index_desc.SetIndexType(IndexTypeCast(value));
                } else if (key == milvus::KeyMetricType()) {
                    index_desc.SetMetricType(MetricTypeCast(value));
                } else if (key == milvus::KeyParams()) {
                    index_desc.ExtraParamsFromJson(value);
                }
            }
        }
    };

    return apiHandler<proto::milvus::DescribeIndexRequest, proto::milvus::DescribeIndexResponse>(
        pre, &MilvusConnection::DescribeIndex, post);
}

Status
MilvusClientImpl::GetIndexState(const std::string& collection_name, const std::string& field_name, IndexState& state) {
    auto pre = [&collection_name, &field_name]() {
        proto::milvus::GetIndexStateRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_field_name(field_name);
        return rpc_request;
    };

    auto post = [&state](const proto::milvus::GetIndexStateResponse& response) {
        state.SetStateCode(IndexStateCast(response.state()));
        state.SetFailedReason(response.fail_reason());
    };

    return apiHandler<proto::milvus::GetIndexStateRequest, proto::milvus::GetIndexStateResponse>(
        pre, &MilvusConnection::GetIndexState, post);
}

Status
MilvusClientImpl::GetIndexBuildProgress(const std::string& collection_name, const std::string& field_name,
                                        IndexProgress& progress) {
    auto pre = [&collection_name, &field_name]() {
        proto::milvus::GetIndexBuildProgressRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_field_name(field_name);
        return rpc_request;
    };

    auto post = [&progress](const proto::milvus::GetIndexBuildProgressResponse& response) {
        progress.SetTotalRows(response.total_rows());
        progress.SetIndexedRows(response.indexed_rows());
    };

    return apiHandler<proto::milvus::GetIndexBuildProgressRequest, proto::milvus::GetIndexBuildProgressResponse>(
        pre, &MilvusConnection::GetIndexBuildProgress, post);
}

Status
MilvusClientImpl::DropIndex(const std::string& collection_name, const std::string& field_name) {
    auto pre = [&collection_name, &field_name]() {
        proto::milvus::DropIndexRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_field_name(field_name);
        return rpc_request;
    };
    return apiHandler<proto::milvus::DropIndexRequest, proto::common::Status>(pre, &MilvusConnection::DropIndex);
}

Status
MilvusClientImpl::Insert(const std::string& collection_name, const std::string& partition_name,
                         const std::vector<FieldDataPtr>& fields, DmlResults& results) {
    // TODO(matrixji): add common validations check for fields
    // TODO(matrixji): add scheme based validations check for fields
    // auto validate = nullptr;

    auto pre = [&collection_name, &partition_name, &fields] {
        proto::milvus::InsertRequest rpc_request;

        auto* mutable_fields = rpc_request.mutable_fields_data();
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_partition_name(partition_name);
        rpc_request.set_num_rows((*fields.front()).Count());
        for (const auto& field : fields) {
            mutable_fields->Add(std::move(CreateProtoFieldData(*field)));
        }
        return rpc_request;
    };

    auto post = [&results](const proto::milvus::MutationResult& response) {
        auto id_array = CreateIDArray(response.ids());
        results.SetIdArray(std::move(id_array));
        results.SetTimestamp(response.timestamp());
    };

    return apiHandler<proto::milvus::InsertRequest, proto::milvus::MutationResult>(pre, &MilvusConnection::Insert,
                                                                                   post);
}

Status
MilvusClientImpl::Delete(const std::string& collection_name, const std::string& partition_name,
                         const std::string& expression, DmlResults& results) {
    auto pre = [&collection_name, &partition_name, &expression]() {
        proto::milvus::DeleteRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_partition_name(partition_name);
        rpc_request.set_expr(expression);
        return rpc_request;
    };

    auto post = [&results](const proto::milvus::MutationResult& response) {
        auto id_array = CreateIDArray(response.ids());
        results.SetIdArray(std::move(id_array));
        results.SetTimestamp(response.timestamp());
    };

    return apiHandler<proto::milvus::DeleteRequest, proto::milvus::MutationResult>(pre, &MilvusConnection::Delete,
                                                                                   post);
}

Status
MilvusClientImpl::Search(const SearchArguments& arguments, SearchResults& results, int timeout) {
    std::string anns_field;
    auto validate = [this, &arguments, &anns_field]() {
        CollectionDesc collection_desc;
        auto status = DescribeCollection(arguments.CollectionName(), collection_desc);
        if (status.IsOk()) {
            // check anns fields
            auto& field_name = arguments.TargetVectors()->Name();
            auto anns_fileds = collection_desc.Schema().AnnsFieldNames();
            if (anns_fileds.find(field_name) != anns_fileds.end()) {
                anns_field = field_name;
            } else {
                return Status{StatusCode::INVALID_AGUMENT, std::string(field_name + " is not a valid anns field")};
            }
            // basic check for extra params
            status = arguments.Validate();
        }
        return status;
    };

    auto pre = [&arguments, &anns_field]() {
        proto::milvus::SearchRequest rpc_request;
        rpc_request.set_collection_name(arguments.CollectionName());
        rpc_request.set_dsl_type(proto::common::DslType::BoolExprV1);
        if (!arguments.Expression().empty()) {
            rpc_request.set_dsl(arguments.Expression());
        }
        for (const auto& partition_name : arguments.PartitionNames()) {
            rpc_request.add_partition_names(partition_name);
        }
        for (const auto& output_field : arguments.OutputFields()) {
            rpc_request.add_output_fields(output_field);
        }

        // placeholders
        proto::common::PlaceholderGroup placeholder_group;
        auto& placeholder_value = *placeholder_group.add_placeholders();
        placeholder_value.set_tag("$0");
        auto target = arguments.TargetVectors();
        if (target->Type() == DataType::BINARY_VECTOR) {
            // bins
            placeholder_value.set_type(proto::common::PlaceholderType::BinaryVector);
            auto& bins_vec = dynamic_cast<BinaryVecFieldData&>(*target);
            for (const auto& bins : bins_vec.Data()) {
                std::string placeholder_data(reinterpret_cast<const char*>(bins.data()), bins.size());
                placeholder_value.add_values(std::move(placeholder_data));
            }
        } else {
            // floats
            placeholder_value.set_type(proto::common::PlaceholderType::FloatVector);
            auto& floats_vec = dynamic_cast<FloatVecFieldData&>(*target);
            for (const auto& floats : floats_vec.Data()) {
                std::string placeholder_data(reinterpret_cast<const char*>(floats.data()),
                                             floats.size() * sizeof(float));
                placeholder_value.add_values(std::move(placeholder_data));
            }
        }
        rpc_request.set_placeholder_group(std::move(placeholder_group.SerializeAsString()));

        auto kv_pair = rpc_request.add_search_params();
        kv_pair->set_key("anns_field");
        kv_pair->set_value(anns_field);

        kv_pair = rpc_request.add_search_params();
        kv_pair->set_key("topk");
        kv_pair->set_value(std::to_string(arguments.TopK()));

        kv_pair = rpc_request.add_search_params();
        kv_pair->set_key(milvus::KeyMetricType());
        kv_pair->set_value(std::to_string(arguments.MetricType()));

        kv_pair = rpc_request.add_search_params();
        kv_pair->set_key("round_decimal");
        kv_pair->set_value(std::to_string(arguments.RoundDecimal()));

        kv_pair = rpc_request.add_search_params();
        kv_pair->set_key(milvus::KeyParams());
        // merge extra params with range search
        auto json = nlohmann::json::parse(arguments.ExtraParams());
        if (arguments.RangeSearch()) {
            json["range_filter"] = arguments.RangeFilter();
            json["radius"] = arguments.Radius();
        }
        kv_pair->set_value(json.dump());

        rpc_request.set_travel_timestamp(arguments.TravelTimestamp());
        rpc_request.set_guarantee_timestamp(arguments.GuaranteeTimestamp());
        return rpc_request;
    };

    auto post = [&results](const proto::milvus::SearchResults& response) {
        auto& result_data = response.results();
        const auto& ids = result_data.ids();
        const auto& scores = result_data.scores();
        const auto& fields_data = result_data.fields_data();
        auto num_of_queries = result_data.num_queries();
        std::vector<int> topks{};
        topks.reserve(result_data.topks_size());
        for (int i = 0; i < result_data.topks_size(); ++i) {
            topks.emplace_back(result_data.topks(i));
        }
        std::vector<SingleResult> single_results;
        single_results.reserve(num_of_queries);
        int offset{0};
        for (int i = 0; i < num_of_queries; ++i) {
            std::vector<float> item_scores;
            std::vector<FieldDataPtr> item_field_data;
            auto item_topk = topks[i];
            item_scores.reserve(item_topk);
            for (int j = 0; j < item_topk; ++j) {
                item_scores.emplace_back(scores.at(offset + j));
            }
            item_field_data.reserve(fields_data.size());
            for (const auto& field_data : fields_data) {
                item_field_data.emplace_back(std::move(milvus::CreateMilvusFieldData(field_data, offset, item_topk)));
            }
            single_results.emplace_back(std::move(CreateIDArray(ids, offset, item_topk)), std::move(item_scores),
                                        std::move(item_field_data));
            offset += item_topk;
        }

        results = std::move(SearchResults(std::move(single_results)));
    };

    return apiHandler<proto::milvus::SearchRequest, proto::milvus::SearchResults>(
        validate, pre, &MilvusConnection::Search, nullptr, post, GrpcOpts{timeout});
}

Status
MilvusClientImpl::Query(const QueryArguments& arguments, QueryResults& results, int timeout) {
    auto pre = [&arguments]() {
        proto::milvus::QueryRequest rpc_request;
        rpc_request.set_collection_name(arguments.CollectionName());
        for (const auto& partition_name : arguments.PartitionNames()) {
            rpc_request.add_partition_names(partition_name);
        }

        rpc_request.set_expr(arguments.Expression());
        for (const auto& field : arguments.OutputFields()) {
            rpc_request.add_output_fields(field);
        }

        rpc_request.set_travel_timestamp(arguments.TravelTimestamp());
        rpc_request.set_guarantee_timestamp(arguments.GuaranteeTimestamp());
        return rpc_request;
    };

    auto post = [&results](const proto::milvus::QueryResults& response) {
        std::vector<milvus::FieldDataPtr> return_fields{};
        return_fields.reserve(response.fields_data_size());
        for (const auto& field_data : response.fields_data()) {
            return_fields.emplace_back(std::move(CreateMilvusFieldData(field_data)));
        }

        results = std::move(QueryResults(std::move(return_fields)));
    };
    return apiHandler<proto::milvus::QueryRequest, proto::milvus::QueryResults>(pre, &MilvusConnection::Query, post,
                                                                                GrpcOpts{timeout});
}

Status
MilvusClientImpl::CalcDistance(const CalcDistanceArguments& arguments, DistanceArray& results) {
    auto validate = [&arguments]() { return arguments.Validate(); };

    auto pre = [&arguments]() {
        auto pass_arguments = [&arguments](proto::milvus::VectorsArray* rpc_vectors, FieldDataPtr&& arg_vectors,
                                           bool is_left) {
            if (IsVectorType(arg_vectors->Type())) {
                auto data_array = rpc_vectors->mutable_data_array();

                if (arg_vectors->Type() == DataType::FLOAT_VECTOR) {
                    FloatVecFieldDataPtr data_ptr = std::static_pointer_cast<FloatVecFieldData>(arg_vectors);
                    auto float_vectors = data_array->mutable_float_vector();
                    auto mutable_data = float_vectors->mutable_data();
                    auto& vectors = data_ptr->Data();
                    for (auto& vector : vectors) {
                        mutable_data->Add(vector.begin(), vector.end());
                    }

                    // suppose vectors is not empty, already checked by Validate()
                    data_array->set_dim(static_cast<int>(vectors[0].size()));
                } else {
                    auto data_ptr = std::static_pointer_cast<BinaryVecFieldData>(arg_vectors);
                    auto& str = *data_array->mutable_binary_vector();
                    auto& vectors = data_ptr->Data();
                    // user specify dimension(only for binary vectors), if not, get it from vectors
                    auto dimensions = arguments.Dimension() > 0 ? arguments.Dimension() : (vectors.front().size() * 8);
                    str.reserve(dimensions * vectors.size() / 8);
                    for (auto& vector : vectors) {
                        str.append(vector);
                    }
                    data_array->set_dim(static_cast<int>(dimensions));
                }

            } else if (arg_vectors->Type() == DataType::INT64) {
                auto id_array = rpc_vectors->mutable_id_array();
                id_array->set_collection_name(is_left ? arguments.LeftCollection() : arguments.RightCollection());
                auto& partitions = is_left ? arguments.LeftPartitions() : arguments.RightPartitions();
                for (auto& name : partitions) {
                    id_array->add_partition_names(name);
                }

                auto ids = id_array->mutable_id_array();
                auto long_ids = ids->mutable_int_id();
                auto mutable_data = long_ids->mutable_data();
                Int64FieldDataPtr data_ptr = std::static_pointer_cast<Int64FieldData>(arg_vectors);
                auto& vector_ids = data_ptr->Data();
                mutable_data->Add(vector_ids.begin(), vector_ids.end());
            }
        };

        // set vectors data
        proto::milvus::CalcDistanceRequest rpc_request;
        pass_arguments(rpc_request.mutable_op_left(), arguments.LeftVectors(), true);
        pass_arguments(rpc_request.mutable_op_right(), arguments.RightVectors(), false);

        // set metric
        auto kv = rpc_request.add_params();
        kv->set_key("metric");
        kv->set_value(arguments.MetricType());

        return std::move(rpc_request);
    };

    auto post = [&arguments, &results](const proto::milvus::CalcDistanceResults& response) {
        size_t left_count = arguments.LeftVectors()->Count();
        size_t right_count = arguments.RightVectors()->Count();
        if (response.has_int_dist()) {
            auto& int_array = response.int_dist();
            auto& distance_array = int_array.data();
            const int32_t* distance_data = distance_array.data();
            auto distance_count = int_array.data_size();
            std::vector<std::vector<int32_t>> all_distances;

            // for id array, suppose all the vectors are exist.
            // if some vectors are missed(or deleted), the server will return error by the CalcDistance api.
            if (distance_count == left_count * right_count) {
                all_distances.reserve(left_count);
                for (size_t i = 0; i < left_count; ++i) {
                    std::vector<int32_t> distances(right_count);
                    memcpy(distances.data(), distance_data + i * right_count, right_count * sizeof(int32_t));
                    all_distances.emplace_back(std::move(distances));
                }
            }
            results.SetIntDistance(std::move(all_distances));
        } else {
            auto& float_array = response.float_dist();
            auto& distance_array = float_array.data();
            const float* distance_data = distance_array.data();
            auto distance_count = float_array.data_size();
            std::vector<std::vector<float>> all_distances;

            // suppose distance count is always equal to left_count * right_count
            if (distance_count == left_count * right_count) {
                all_distances.reserve(left_count);
                for (size_t i = 0; i < left_count; ++i) {
                    std::vector<float> distances(right_count);
                    memcpy(distances.data(), distance_data + i * right_count, right_count * sizeof(float));
                    all_distances.emplace_back(std::move(distances));
                }
            }
            results.SetFloatDistance(std::move(all_distances));
        }
    };
    return apiHandler<proto::milvus::CalcDistanceRequest, proto::milvus::CalcDistanceResults>(
        validate, pre, &MilvusConnection::CalcDistance, post);
}

Status
MilvusClientImpl::Flush(const std::vector<std::string>& collection_names, const ProgressMonitor& progress_monitor) {
    auto pre = [&collection_names]() {
        proto::milvus::FlushRequest rpc_request;
        for (const auto& collection_name : collection_names) {
            rpc_request.add_collection_names(collection_name);
        }
        return rpc_request;
    };

    auto wait_for_status = [this, &progress_monitor](const proto::milvus::FlushResponse& response) {
        std::map<std::string, std::vector<int64_t>> flush_segments;
        for (const auto& iter : response.coll_segids()) {
            const auto& ids = iter.second.data();
            std::vector<int64_t> seg_ids;
            seg_ids.reserve(ids.size());
            seg_ids.insert(seg_ids.end(), ids.begin(), ids.end());
            flush_segments.insert(std::make_pair(iter.first, seg_ids));
        }

        // the segment_count is how many segments need to be flushed
        // the finished_count is how many segments have been flushed
        uint32_t segment_count = 0, finished_count = 0;
        for (auto& pair : flush_segments) {
            segment_count += pair.second.size();
        }
        if (segment_count == 0) {
            return Status::OK();
        }

        return WaitForStatus(
            [&segment_count, &flush_segments, &finished_count, this](Progress& p) -> Status {
                p.total_ = segment_count;

                // call GetFlushState() to check segment state
                for (auto iter = flush_segments.begin(); iter != flush_segments.end();) {
                    bool flushed = false;
                    Status status = GetFlushState(iter->second, flushed);
                    if (!status.IsOk()) {
                        return status;
                    }

                    if (flushed) {
                        finished_count += iter->second.size();
                        flush_segments.erase(iter++);
                    } else {
                        iter++;
                    }
                }
                p.finished_ = finished_count;

                return Status::OK();
            },
            progress_monitor);
    };

    return apiHandler<proto::milvus::FlushRequest, proto::milvus::FlushResponse>(nullptr, pre, &MilvusConnection::Flush,
                                                                                 wait_for_status, nullptr);
}

Status
MilvusClientImpl::GetFlushState(const std::vector<int64_t>& segments, bool& flushed) {
    auto pre = [&segments]() {
        proto::milvus::GetFlushStateRequest rpc_request;
        for (auto id : segments) {
            rpc_request.add_segmentids(id);
        }
        return rpc_request;
    };

    auto post = [&flushed](const proto::milvus::GetFlushStateResponse& response) { flushed = response.flushed(); };

    return apiHandler<proto::milvus::GetFlushStateRequest, proto::milvus::GetFlushStateResponse>(
        pre, &MilvusConnection::GetFlushState, post);
}

Status
MilvusClientImpl::GetPersistentSegmentInfo(const std::string& collection_name, SegmentsInfo& segments_info) {
    auto pre = [&collection_name] {
        proto::milvus::GetPersistentSegmentInfoRequest rpc_request;
        rpc_request.set_collectionname(collection_name);
        return rpc_request;
    };

    auto post = [&segments_info](const proto::milvus::GetPersistentSegmentInfoResponse& response) {
        for (const auto& info : response.infos()) {
            segments_info.emplace_back(info.collectionid(), info.partitionid(), info.segmentid(), info.num_rows(),
                                       SegmentStateCast(info.state()));
        }
    };

    return apiHandler<proto::milvus::GetPersistentSegmentInfoRequest, proto::milvus::GetPersistentSegmentInfoResponse>(
        pre, &MilvusConnection::GetPersistentSegmentInfo, post);
}

Status
MilvusClientImpl::GetQuerySegmentInfo(const std::string& collection_name, QuerySegmentsInfo& segments_info) {
    auto pre = [&collection_name] {
        proto::milvus::GetQuerySegmentInfoRequest rpc_request;
        rpc_request.set_collectionname(collection_name);
        return rpc_request;
    };

    auto post = [&segments_info](const proto::milvus::GetQuerySegmentInfoResponse& response) {
        for (const auto& info : response.infos()) {
            segments_info.emplace_back(info.collectionid(), info.partitionid(), info.segmentid(), info.num_rows(),
                                       milvus::SegmentStateCast(info.state()), info.index_name(), info.indexid(),
                                       info.nodeid());
        }
    };
    return apiHandler<proto::milvus::GetQuerySegmentInfoRequest, proto::milvus::GetQuerySegmentInfoResponse>(
        pre, &MilvusConnection::GetQuerySegmentInfo, post);
}

Status
MilvusClientImpl::GetMetrics(const std::string& request, std::string& response, std::string& component_name) {
    auto pre = [&request]() {
        proto::milvus::GetMetricsRequest rpc_request;
        rpc_request.set_request(request);
        return rpc_request;
    };

    auto post = [&response, &component_name](const proto::milvus::GetMetricsResponse& rpc_response) {
        response = rpc_response.response();
        component_name = rpc_response.component_name();
    };

    return apiHandler<proto::milvus::GetMetricsRequest, proto::milvus::GetMetricsResponse>(
        pre, &MilvusConnection::GetMetrics, post);
}

Status
MilvusClientImpl::LoadBalance(int64_t src_node, const std::vector<int64_t>& dst_nodes,
                              const std::vector<int64_t>& segments) {
    auto pre = [src_node, &dst_nodes, &segments] {
        proto::milvus::LoadBalanceRequest rpc_request;
        rpc_request.set_src_nodeid(src_node);
        for (const auto dst_node : dst_nodes) {
            rpc_request.add_dst_nodeids(dst_node);
        }
        for (const auto segment : segments) {
            rpc_request.add_sealed_segmentids(segment);
        }
        return rpc_request;
    };

    return apiHandler<proto::milvus::LoadBalanceRequest, proto::common::Status>(pre, &MilvusConnection::LoadBalance,
                                                                                nullptr);
}

Status
MilvusClientImpl::GetCompactionState(int64_t compaction_id, CompactionState& compaction_state) {
    auto pre = [&compaction_id]() {
        proto::milvus::GetCompactionStateRequest rpc_request;
        rpc_request.set_compactionid(compaction_id);
        return rpc_request;
    };

    auto post = [&compaction_state](const proto::milvus::GetCompactionStateResponse& response) {
        compaction_state.SetExecutingPlan(response.executingplanno());
        compaction_state.SetTimeoutPlan(response.timeoutplanno());
        compaction_state.SetCompletedPlan(response.completedplanno());
        switch (response.state()) {
            case proto::common::CompactionState::Completed:
                compaction_state.SetState(CompactionStateCode::COMPLETED);
                break;
            case proto::common::CompactionState::Executing:
                compaction_state.SetState(CompactionStateCode::EXECUTING);
                break;
            default:
                break;
        }
    };

    return apiHandler<proto::milvus::GetCompactionStateRequest, proto::milvus::GetCompactionStateResponse>(
        pre, &MilvusConnection::GetCompactionState, post);
}

Status
MilvusClientImpl::ManualCompaction(const std::string& collection_name, uint64_t travel_timestamp,
                                   int64_t& compaction_id) {
    CollectionDesc collection_desc;
    auto status = DescribeCollection(collection_name, collection_desc);
    if (!status.IsOk()) {
        return status;
    }

    auto pre = [&travel_timestamp, &collection_desc]() {
        proto::milvus::ManualCompactionRequest rpc_request;
        rpc_request.set_collectionid(collection_desc.ID());
        rpc_request.set_timetravel(travel_timestamp);
        return rpc_request;
    };

    auto post = [&compaction_id](const proto::milvus::ManualCompactionResponse& response) {
        compaction_id = response.compactionid();
    };

    return apiHandler<proto::milvus::ManualCompactionRequest, proto::milvus::ManualCompactionResponse>(
        pre, &MilvusConnection::ManualCompaction, post);
}

Status
MilvusClientImpl::GetCompactionPlans(int64_t compaction_id, CompactionPlans& plans) {
    auto pre = [&compaction_id]() {
        proto::milvus::GetCompactionPlansRequest rpc_request;
        rpc_request.set_compactionid(compaction_id);
        return rpc_request;
    };

    auto post = [&plans](const proto::milvus::GetCompactionPlansResponse& response) {
        for (int i = 0; i < response.mergeinfos_size(); ++i) {
            auto& info = response.mergeinfos(i);
            std::vector<int64_t> source_ids;
            source_ids.reserve(info.sources_size());
            source_ids.insert(source_ids.end(), info.sources().begin(), info.sources().end());
            plans.emplace_back(source_ids, info.target());
        }
    };

    return apiHandler<proto::milvus::GetCompactionPlansRequest, proto::milvus::GetCompactionPlansResponse>(
        pre, &MilvusConnection::GetCompactionPlans, post);
}

Status
MilvusClientImpl::CreateCredential(const std::string& username, const std::string& password) {
    auto pre = [&username, &password]() {
        proto::milvus::CreateCredentialRequest rpc_request;
        rpc_request.set_username(username);
        rpc_request.set_password(milvus::Base64Encode(password));
        // TODO: seconds or milliseconds?
        auto timestamp =
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
                .count();
        rpc_request.set_created_utc_timestamps(timestamp);
        rpc_request.set_modified_utc_timestamps(timestamp);
        return rpc_request;
    };

    return apiHandler<proto::milvus::CreateCredentialRequest, proto::common::Status>(
        pre, &MilvusConnection::CreateCredential, nullptr);
}

Status
MilvusClientImpl::UpdateCredential(const std::string& username, const std::string& old_password,
                                   const std::string& new_password) {
    auto pre = [&username, &old_password, &new_password]() {
        proto::milvus::UpdateCredentialRequest rpc_request;
        rpc_request.set_username(username);
        rpc_request.set_oldpassword(milvus::Base64Encode(old_password));
        rpc_request.set_newpassword(milvus::Base64Encode(new_password));
        // TODO: seconds or milliseconds?
        rpc_request.set_modified_utc_timestamps(
            std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch())
                .count());
        return rpc_request;
    };

    return apiHandler<proto::milvus::UpdateCredentialRequest, proto::common::Status>(
        pre, &MilvusConnection::UpdateCredential, nullptr);
}

Status
MilvusClientImpl::DeleteCredential(const std::string& username) {
    auto pre = [&username]() {
        proto::milvus::DeleteCredentialRequest rpc_request;
        rpc_request.set_username(username);
        return rpc_request;
    };

    return apiHandler<proto::milvus::DeleteCredentialRequest, proto::common::Status>(
        pre, &MilvusConnection::DeleteCredential, nullptr);
}

Status
MilvusClientImpl::ListCredUsers(std::vector<std::string>& users) {
    auto pre = []() {
        proto::milvus::ListCredUsersRequest rpc_request;
        return rpc_request;
    };

    auto post = [&users](const proto::milvus::ListCredUsersResponse& response) {
        users.clear();
        for (const auto& user : response.usernames()) {
            users.emplace_back(user);
        }
    };

    return apiHandler<proto::milvus::ListCredUsersRequest, proto::milvus::ListCredUsersResponse>(
        pre, &MilvusConnection::ListCredUsers, post);
}

Status
MilvusClientImpl::WaitForStatus(const std::function<Status(Progress&)>& query_function,
                                const ProgressMonitor& progress_monitor) {
    // no need to check
    if (progress_monitor.CheckTimeout() == 0) {
        return Status::OK();
    }

    std::chrono::time_point<std::chrono::steady_clock> started = std::chrono::steady_clock::now();

    auto calculated_next_wait = started;
    auto wait_milliseconds = progress_monitor.CheckTimeout() * 1000;
    auto wait_interval = progress_monitor.CheckInterval();
    auto final_timeout = started + std::chrono::milliseconds{wait_milliseconds};
    while (true) {
        calculated_next_wait += std::chrono::milliseconds{wait_interval};
        auto next_wait = std::min(calculated_next_wait, final_timeout);
        std::this_thread::sleep_until(next_wait);

        Progress current_progress;
        auto status = query_function(current_progress);

        // if the internal check function failed, return error
        if (!status.IsOk()) {
            return status;
        }

        // notify progress
        progress_monitor.DoProgress(current_progress);

        // if progress all done, break the circle
        if (current_progress.Done()) {
            return status;
        }

        // if time to deadline, return timeout error
        if (next_wait >= final_timeout) {
            return Status{StatusCode::TIMEOUT, "time out"};
        }
    }
}

}  // namespace milvus
