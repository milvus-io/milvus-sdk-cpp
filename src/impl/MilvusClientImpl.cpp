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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <thread>

#include "common.pb.h"
#include "milvus.pb.h"
#include "schema.pb.h"
#include "utils/Constants.h"
#include "utils/DmlUtils.h"
#include "utils/GtsDict.h"
#include "utils/TypeUtils.h"

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
        rpc_collection.set_enable_dynamic_field(schema.EnableDynamicField());

        for (auto& field : schema.Fields()) {
            proto::schema::FieldSchema* rpc_field = rpc_collection.add_fields();
            rpc_field->set_name(field.Name());
            rpc_field->set_description(field.Description());
            rpc_field->set_data_type(static_cast<proto::schema::DataType>(field.FieldDataType()));
            rpc_field->set_is_primary_key(field.IsPrimaryKey());
            rpc_field->set_autoid(field.AutoID());

            if (field.FieldDataType() == DataType::ARRAY) {
                rpc_field->set_element_type(static_cast<proto::schema::DataType>(field.ElementType()));
            }

            for (auto& pair : field.TypeParams()) {
                proto::common::KeyValuePair* kv = rpc_field->add_type_params();
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

    auto post = [this, &collection_name](const proto::common::Status& status) {
        if (status.error_code() == proto::common::ErrorCode::Success && status.code() == 0) {
            // TODO: if the parameters provides db_name in future, we need to set the correct
            // db_name to RemoveCollectionTs()
            GtsDict::GetInstance().RemoveCollectionTs(currentDbName(""), collection_name);
            removeCollectionDesc(collection_name);
        }
    };

    return apiHandler<proto::milvus::DropCollectionRequest, proto::common::Status>(
        pre, &MilvusConnection::DropCollection, post);
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
        pre, &MilvusConnection::GetCollectionStatistics, post);
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
        pre, &MilvusConnection::GetPartitionStatistics, post);
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
MilvusClientImpl::UseDatabase(const std::string& db_name) {
    cleanCollectionDescCache();
    if (connection_ != nullptr) {
        return connection_->UseDatabase(db_name);
    }

    return Status::OK();
}

Status
MilvusClientImpl::CreateDatabase(const std::string& db_name,
                                 const std::unordered_map<std::string, std::string>& properties) {
    auto pre = [&db_name, &properties]() {
        proto::milvus::CreateDatabaseRequest rpc_request;
        rpc_request.set_db_name(db_name);

        for (const auto& pair : properties) {
            auto kv_pair = rpc_request.add_properties();
            kv_pair->set_key(pair.first);
            kv_pair->set_value(pair.second);
        }

        return rpc_request;
    };

    return apiHandler<proto::milvus::CreateDatabaseRequest, proto::common::Status>(pre,
                                                                                   &MilvusConnection::CreateDatabase);
}

Status
MilvusClientImpl::DropDatabase(const std::string& db_name) {
    auto pre = [&db_name]() {
        proto::milvus::DropDatabaseRequest rpc_request;
        rpc_request.set_db_name(db_name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::DropDatabaseRequest, proto::common::Status>(pre, &MilvusConnection::DropDatabase);
}

Status
MilvusClientImpl::ListDatabases(std::vector<std::string>& names) {
    auto pre = []() {
        proto::milvus::ListDatabasesRequest rpc_request;
        return rpc_request;
    };

    auto post = [&names](const proto::milvus::ListDatabasesResponse& response) {
        for (int i = 0; i < response.db_names_size(); i++) {
            names.push_back(response.db_names(i));
        }
    };

    return apiHandler<proto::milvus::ListDatabasesRequest, proto::milvus::ListDatabasesResponse>(
        pre, &MilvusConnection::ListDatabases, post);
}

Status
MilvusClientImpl::AlterDatabaseProperties(const std::string& db_name,
                                          const std::unordered_map<std::string, std::string>& properties) {
    auto pre = [&db_name, &properties]() {
        proto::milvus::AlterDatabaseRequest rpc_request;
        rpc_request.set_db_name(db_name);

        for (const auto& pair : properties) {
            auto kv_pair = rpc_request.add_properties();
            kv_pair->set_key(pair.first);
            kv_pair->set_value(pair.second);
        }

        return rpc_request;
    };

    return apiHandler<proto::milvus::AlterDatabaseRequest, proto::common::Status>(pre,
                                                                                  &MilvusConnection::AlterDatabase);
}

Status
MilvusClientImpl::DropDatabaseProperties(const std::string& db_name, const std::vector<std::string>& properties) {
    auto pre = [&db_name, &properties]() {
        proto::milvus::AlterDatabaseRequest rpc_request;
        rpc_request.set_db_name(db_name);

        for (const auto& name : properties) {
            rpc_request.add_delete_keys(name);
        }

        return rpc_request;
    };

    return apiHandler<proto::milvus::AlterDatabaseRequest, proto::common::Status>(pre,
                                                                                  &MilvusConnection::AlterDatabase);
}

Status
MilvusClientImpl::DescribeDatabase(const std::string& db_name, DatabaseDesc& db_desc) {
    auto pre = [&db_name]() {
        proto::milvus::DescribeDatabaseRequest rpc_request;
        rpc_request.set_db_name(db_name);
        return rpc_request;
    };

    auto post = [&db_desc](const proto::milvus::DescribeDatabaseResponse& response) {
        db_desc.SetName(response.db_name());
        db_desc.SetID(response.dbid());
        db_desc.SetCreatedTime(response.created_timestamp());
        std::unordered_map<std::string, std::string> properties;
        for (int i = 0; i < response.properties_size(); i++) {
            const auto& prop = response.properties(i);
            properties[prop.key()] = prop.value();
        }
        db_desc.SetProperties(properties);
    };

    return apiHandler<proto::milvus::DescribeDatabaseRequest, proto::milvus::DescribeDatabaseResponse>(
        pre, &MilvusConnection::DescribeDatabase, post);
}

Status
MilvusClientImpl::CreateIndex(const std::string& collection_name, const IndexDesc& index_desc,
                              const ProgressMonitor& progress_monitor) {
    auto pre = [&collection_name, &index_desc]() {
        proto::milvus::CreateIndexRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_field_name(index_desc.FieldName());
        rpc_request.set_index_name(index_desc.IndexName());

        auto kv_pair = rpc_request.add_extra_params();
        kv_pair->set_key(milvus::KeyIndexType());
        kv_pair->set_value(std::to_string(index_desc.IndexType()));

        // for scalar fields, no metric type
        if (index_desc.MetricType() != MetricType::DEFAULT) {
            kv_pair = rpc_request.add_extra_params();
            kv_pair->set_key(milvus::KeyMetricType());
            kv_pair->set_value(std::to_string(index_desc.MetricType()));
        }

        kv_pair = rpc_request.add_extra_params();
        kv_pair->set_key(milvus::KeyParams());
        ::nlohmann::json json_obj(index_desc.ExtraParams());
        kv_pair->set_value(json_obj.dump());

        return rpc_request;
    };

    auto wait_for_status = [&collection_name, &index_desc, &progress_monitor, this](const proto::common::Status&) {
        return WaitForStatus(
            [&collection_name, &index_desc, this](Progress& progress) -> Status {
                IndexDesc index_state;
                auto status = DescribeIndex(collection_name, index_desc.FieldName(), index_state);
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
                    return Status{StatusCode::SERVER_FAILED, "index failed:" + index_state.FailReason()};
                }

                return status;
            },
            progress_monitor);
    };
    return apiHandler<proto::milvus::CreateIndexRequest, proto::common::Status>(
        nullptr, pre, &MilvusConnection::CreateIndex, wait_for_status, nullptr);
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

    auto post = [&index_desc, &field_name](const proto::milvus::DescribeIndexResponse& response) {
        auto count = response.index_descriptions_size();
        int poz = -1;
        if (count == 1) {
            poz = 0;
        } else {
            for (int i = 0; i < count; ++i) {
                auto rpc_desc = response.index_descriptions(i);
                if (field_name == rpc_desc.field_name()) {
                    poz = i;
                    break;
                }
            }
        }

        if (poz >= 0) {
            auto rpc_desc = response.index_descriptions(poz);
            index_desc.SetFieldName(rpc_desc.field_name());
            index_desc.SetIndexName(rpc_desc.index_name());
            index_desc.SetIndexId(rpc_desc.indexid());
            index_desc.SetStateCode(IndexStateCast(rpc_desc.state()));
            index_desc.SetFailReason(rpc_desc.index_state_fail_reason());
            index_desc.SetIndexedRows(rpc_desc.indexed_rows());
            index_desc.SetTotalRows(rpc_desc.total_rows());
            index_desc.SetPendingRows(rpc_desc.pending_index_rows());
            auto index_params_size = rpc_desc.params_size();
            for (int j = 0; j < index_params_size; ++j) {
                const auto& key = rpc_desc.params(j).key();
                const auto& value = rpc_desc.params(j).value();
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
MilvusClientImpl::DropIndex(const std::string& collection_name, const std::string& index_name) {
    auto pre = [&collection_name, &index_name]() {
        proto::milvus::DropIndexRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_index_name(index_name);
        return rpc_request;
    };
    return apiHandler<proto::milvus::DropIndexRequest, proto::common::Status>(pre, &MilvusConnection::DropIndex);
}

Status
MilvusClientImpl::Insert(const std::string& collection_name, const std::string& partition_name,
                         const std::vector<FieldDataPtr>& fields, DmlResults& results) {
    bool enable_dynamic_field;
    auto validate = [this, &collection_name, &fields, &enable_dynamic_field]() {
        CollectionDescPtr collection_desc;
        auto status = getCollectionDesc(collection_name, false, collection_desc);
        if (!status.IsOk()) {
            return status;
        }

        // if the collection is already recreated, some schema might be changed, we need to update the
        // collectionDesc cache and call CheckInsertInput() again.
        status = CheckInsertInput(collection_desc, fields, false);
        if (status.Code() == milvus::StatusCode::DATA_UNMATCH_SCHEMA) {
            status = getCollectionDesc(collection_name, true, collection_desc);
            if (!status.IsOk()) {
                return status;
            }

            status = CheckInsertInput(collection_desc, fields, false);
        }

        return status;
    };

    auto pre = [&collection_name, &partition_name, &fields, &enable_dynamic_field] {
        proto::milvus::InsertRequest rpc_request;

        auto* mutable_fields = rpc_request.mutable_fields_data();
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_partition_name(partition_name);
        rpc_request.set_num_rows((*fields.front()).Count());
        for (const auto& field : fields) {
            proto::schema::FieldData data = CreateProtoFieldData(*field);
            if (enable_dynamic_field && field->Name() == DynamicFieldName()) {
                data.set_is_dynamic(true);
            }
            mutable_fields->Add(std::move(data));
        }
        return rpc_request;
    };

    auto post = [this, &collection_name, &results](const proto::milvus::MutationResult& response) {
        auto id_array = CreateIDArray(response.ids());
        results.SetIdArray(std::move(id_array));
        results.SetTimestamp(response.timestamp());

        // special for dml api: if the api failed, remove the schema cache of this collection
        if (IsRealFailure(response.status())) {
            removeCollectionDesc(collection_name);
        } else {
            // TODO: if the parameters provides db_name in future, we need to set the correct
            // db_name to UpdateCollectionTs()
            GtsDict::GetInstance().UpdateCollectionTs(currentDbName(""), collection_name, response.timestamp());
        }
    };

    auto status = apiHandler<proto::milvus::InsertRequest, proto::milvus::MutationResult>(
        validate, pre, &MilvusConnection::Insert, post);
    // If there are multiple clients, the client_A repeatedly do insert, the client_B changes
    // the collection schema. The server might return a special error code "SchemaMismatch".
    // If the client_A gets this special error code, it needs to update the collectionDesc cache and
    // call Insert() again.
    if (status.LegacyServerCode() == static_cast<int32_t>(proto::common::ErrorCode::SchemaMismatch)) {
        removeCollectionDesc(collection_name);
        return Insert(collection_name, partition_name, fields, results);
    }
    return status;
}

Status
MilvusClientImpl::Upsert(const std::string& collection_name, const std::string& partition_name,
                         const std::vector<FieldDataPtr>& fields, DmlResults& results) {
    bool enable_dynamic_field;
    auto validate = [this, &collection_name, &fields, &enable_dynamic_field]() {
        CollectionDescPtr collection_desc;
        auto status = getCollectionDesc(collection_name, false, collection_desc);
        if (!status.IsOk()) {
            return status;
        }

        // if the collection is already recreated, some schema might be changed, we need to update the
        // collectionDesc cache and call CheckInsertInput() again.
        status = CheckInsertInput(collection_desc, fields, true);
        if (status.Code() == milvus::StatusCode::DATA_UNMATCH_SCHEMA) {
            status = getCollectionDesc(collection_name, true, collection_desc);
            if (!status.IsOk()) {
                return status;
            }

            status = CheckInsertInput(collection_desc, fields, true);
        }
        return status;
    };

    auto pre = [&collection_name, &partition_name, &fields, &enable_dynamic_field] {
        proto::milvus::UpsertRequest rpc_request;

        auto* mutable_fields = rpc_request.mutable_fields_data();
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_partition_name(partition_name);
        rpc_request.set_num_rows((*fields.front()).Count());
        for (const auto& field : fields) {
            proto::schema::FieldData data = CreateProtoFieldData(*field);
            if (enable_dynamic_field && field->Name() == DynamicFieldName()) {
                data.set_is_dynamic(true);
            }
            mutable_fields->Add(std::move(data));
        }
        return rpc_request;
    };

    auto post = [this, &collection_name, &results](const proto::milvus::MutationResult& response) {
        auto id_array = CreateIDArray(response.ids());
        results.SetIdArray(std::move(id_array));
        results.SetTimestamp(response.timestamp());

        // special for dml api: if the api failed, remove the schema cache of this collection
        if (IsRealFailure(response.status())) {
            removeCollectionDesc(collection_name);
        } else {
            // TODO: if the parameters provides db_name in future, we need to set the correct
            // db_name to UpdateCollectionTs()
            GtsDict::GetInstance().UpdateCollectionTs(currentDbName(""), collection_name, response.timestamp());
        }
    };

    auto status = apiHandler<proto::milvus::UpsertRequest, proto::milvus::MutationResult>(
        validate, pre, &MilvusConnection::Upsert, post);
    // If there are multiple clients, the client_A repeatedly do insert, the client_B changes
    // the collection schema. The server might return a special error code "SchemaMismatch".
    // If the client_A gets this special error code, it needs to update the collectionDesc cache and
    // call Upsert() again.
    if (status.LegacyServerCode() == static_cast<int32_t>(proto::common::ErrorCode::SchemaMismatch)) {
        removeCollectionDesc(collection_name);
        return Upsert(collection_name, partition_name, fields, results);
    }
    return status;
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

    auto post = [this, &results, &collection_name](const proto::milvus::MutationResult& response) {
        auto id_array = CreateIDArray(response.ids());
        results.SetIdArray(std::move(id_array));
        results.SetTimestamp(response.timestamp());

        if (!IsRealFailure(response.status())) {
            // TODO: if the parameters provides db_name in future, we need to set the correct
            // db_name to UpdateCollectionTs()
            GtsDict::GetInstance().UpdateCollectionTs(currentDbName(""), collection_name, response.timestamp());
        }
    };

    return apiHandler<proto::milvus::DeleteRequest, proto::milvus::MutationResult>(pre, &MilvusConnection::Delete,
                                                                                   post);
}

Status
MilvusClientImpl::Search(const SearchArguments& arguments, SearchResults& results, int timeout) {
    auto validate = [&arguments]() { return arguments.Validate(); };

    auto pre = [this, &arguments]() {
        proto::milvus::SearchRequest rpc_request;
        auto db_name = arguments.DatabaseName();
        if (!db_name.empty()) {
            rpc_request.set_db_name(db_name);
        }
        rpc_request.set_collection_name(arguments.CollectionName());
        rpc_request.set_dsl_type(proto::common::DslType::BoolExprV1);
        if (!arguments.Filter().empty()) {
            rpc_request.set_dsl(arguments.Filter());
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
            auto& vectors = dynamic_cast<BinaryVecFieldData&>(*target);
            for (const auto& bins : vectors.Data()) {
                std::string placeholder_data(reinterpret_cast<const char*>(bins.data()), bins.size());
                placeholder_value.add_values(std::move(placeholder_data));
            }
            rpc_request.set_nq(static_cast<int64_t>(vectors.Count()));
        } else if (target->Type() == DataType::FLOAT_VECTOR) {
            // floats
            placeholder_value.set_type(proto::common::PlaceholderType::FloatVector);
            auto& vectors = dynamic_cast<FloatVecFieldData&>(*target);
            for (const auto& floats : vectors.Data()) {
                std::string placeholder_data(reinterpret_cast<const char*>(floats.data()),
                                             floats.size() * sizeof(float));
                placeholder_value.add_values(std::move(placeholder_data));
            }
            rpc_request.set_nq(static_cast<int64_t>(vectors.Count()));
        } else if (target->Type() == DataType::SPARSE_FLOAT_VECTOR) {
            // sparse
            placeholder_value.set_type(proto::common::PlaceholderType::SparseFloatVector);
            auto& vectors = dynamic_cast<SparseFloatVecFieldData&>(*target);
            for (const auto& sparse : vectors.Data()) {
                std::string placeholder_data = EncodeSparseFloatVector(sparse);
                placeholder_value.add_values(std::move(placeholder_data));
            }
            rpc_request.set_nq(static_cast<int64_t>(vectors.Count()));
        }
        rpc_request.set_placeholder_group(std::move(placeholder_group.SerializeAsString()));

        // set anns field name, if the name is empty and the collection has only one vector field,
        // milvus server will use the vector field name as anns name. If the collection has multiple
        // vector fields, user needs to explicitly provide an anns field name.
        auto anns_field = arguments.AnnsField();
        if (!anns_field.empty()) {
            auto kv_pair = rpc_request.add_search_params();
            kv_pair->set_key(milvus::KeyAnnsField());
            kv_pair->set_value(anns_field);
        }

        // for history reason, query() requires "limit", search() requires "topk"
        {
            auto kv_pair = rpc_request.add_search_params();
            kv_pair->set_key(milvus::KeyTopK());
            kv_pair->set_value(std::to_string(arguments.Limit()));
        }

        // set this value only when client specified, otherwise let server to get it from index parameters
        if (arguments.MetricType() != MetricType::DEFAULT) {
            auto kv_pair = rpc_request.add_search_params();
            kv_pair->set_key(milvus::KeyMetricType());
            kv_pair->set_value(std::to_string(arguments.MetricType()));
        }

        // round decimal
        {
            auto kv_pair = rpc_request.add_search_params();
            kv_pair->set_key(KeyRoundDecimal());
            kv_pair->set_value(std::to_string(arguments.RoundDecimal()));
        }

        // ignore growing
        {
            auto kv_pair = rpc_request.add_search_params();
            kv_pair->set_key(KeyIgnoreGrowing());
            kv_pair->set_value(arguments.IgnoreGrowing() ? "true" : "false");
        }

        // offet/radius/range_filter/nprobe etc.
        // in old milvus versions, all extra params are under "params"
        // in new milvus versions, all extra params are in the top level
        auto& params = arguments.ExtraParams();
        nlohmann::json json_params;
        for (auto& pair : params) {
            auto kv_pair = rpc_request.add_search_params();
            kv_pair->set_key(pair.first);
            kv_pair->set_value(pair.second);

            // for radius/range, the value should be a numeric instead a string in the JSON string
            // for example:
            //   '{"radius": "2.5", "range_filter": "0.5"}' is illegal in the server-side
            //   '{"radius": 2.5, "range_filter": 0.5}' is ok
            if (pair.first == KeyRadius() || pair.first == KeyRangeFilter()) {
                json_params[pair.first] = atof(pair.second.c_str());
            } else {
                json_params[pair.first] = pair.second;
            }
        }
        {
            auto kv_pair = rpc_request.add_search_params();
            kv_pair->set_key(KeyParams());
            kv_pair->set_value(json_params.dump());
        }

        ConsistencyLevel level = arguments.GetConsistencyLevel();
        uint64_t guarantee_ts = DeduceGuaranteeTimestamp(level, currentDbName(db_name), arguments.CollectionName());
        rpc_request.set_guarantee_timestamp(guarantee_ts);
        rpc_request.set_travel_timestamp(arguments.TravelTimestamp());

        if (level == ConsistencyLevel::NONE) {
            rpc_request.set_use_default_consistency(true);
        } else {
            rpc_request.set_consistency_level(ConsistencyLevelCast(arguments.GetConsistencyLevel()));
        }

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
            single_results.emplace_back(result_data.primary_field_name(),
                                        std::move(CreateIDArray(ids, offset, item_topk)), std::move(item_scores),
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
    auto pre = [this, &arguments]() {
        proto::milvus::QueryRequest rpc_request;
        auto db_name = arguments.DatabaseName();
        if (!db_name.empty()) {
            rpc_request.set_db_name(db_name);
        }
        rpc_request.set_collection_name(arguments.CollectionName());
        for (const auto& partition_name : arguments.PartitionNames()) {
            rpc_request.add_partition_names(partition_name);
        }

        rpc_request.set_expr(arguments.Filter());
        for (const auto& field : arguments.OutputFields()) {
            rpc_request.add_output_fields(field);
        }

        // limit/offet etc.
        auto& params = arguments.ExtraParams();
        for (auto& pair : params) {
            auto kv_pair = rpc_request.add_query_params();
            kv_pair->set_key(pair.first);
            kv_pair->set_value(pair.second);
        }

        ConsistencyLevel level = arguments.GetConsistencyLevel();
        uint64_t guarantee_ts = DeduceGuaranteeTimestamp(level, currentDbName(db_name), arguments.CollectionName());
        rpc_request.set_guarantee_timestamp(guarantee_ts);
        rpc_request.set_travel_timestamp(arguments.TravelTimestamp());

        if (level == ConsistencyLevel::NONE) {
            rpc_request.set_use_default_consistency(true);
        } else {
            rpc_request.set_consistency_level(ConsistencyLevelCast(arguments.GetConsistencyLevel()));
        }

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
MilvusClientImpl::GetLoadState(const std::string& collection_name, bool& is_loaded,
                               const std::vector<std::string> partition_names, int timeout) {
    auto pre = [&collection_name, &partition_names]() {
        proto::milvus::GetLoadStateRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        for (const auto& partition_name : partition_names) {
            rpc_request.add_partition_names(partition_name);
        }
        return rpc_request;
    };

    auto post = [&is_loaded](const proto::milvus::GetLoadStateResponse& response) {
        is_loaded = response.state() == proto::common::LoadStateLoaded;
    };

    return apiHandler<proto::milvus::GetLoadStateRequest, proto::milvus::GetLoadStateResponse>(
        pre, &MilvusConnection::GetLoadState, post, GrpcOpts{timeout});
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// internal used methods
Status
MilvusClientImpl::getCollectionDesc(const std::string& collection_name, bool forceUpdate, CollectionDescPtr& descPtr) {
    // this lock locks the entire section, including the call of DescribeCollection()
    // the reason is: describeCollection() could be limited by server-side(DDL request throttling is enabled)
    // we don't intend to allow too many threads run into describeCollection() in this method
    std::lock_guard<std::mutex> lock(collection_desc_cache_mtx_);
    auto it = collection_desc_cache_.find(collection_name);
    if (it != collection_desc_cache_.end()) {
        if (it->second != nullptr && !forceUpdate) {
            descPtr = it->second;
            return Status::OK();
        }
    }

    CollectionDesc desc;
    auto status = DescribeCollection(collection_name, desc);
    if (status.IsOk()) {
        descPtr = std::make_shared<CollectionDesc>(desc);
        collection_desc_cache_[collection_name] = descPtr;
        return status;
    }
    return status;
}

void
MilvusClientImpl::cleanCollectionDescCache() {
    std::lock_guard<std::mutex> lock(collection_desc_cache_mtx_);
    collection_desc_cache_.clear();
}

void
MilvusClientImpl::removeCollectionDesc(const std::string& collection_name) {
    std::lock_guard<std::mutex> lock(collection_desc_cache_mtx_);
    collection_desc_cache_.erase(collection_name);
}

std::string
MilvusClientImpl::currentDbName(const std::string& overwrite_db_name) const {
    // if a db name is specified for rpc interface, use this name
    if (!overwrite_db_name.empty()) {
        return overwrite_db_name;
    }
    // no db name is specified, use the current db name used by this connection
    if (connection_ != nullptr) {
        const ConnectParam& param = connection_->GetConnectParam();
        return param.DbName();
    }
    return "";
}

}  // namespace milvus
