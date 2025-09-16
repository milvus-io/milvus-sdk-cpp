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
#include <type_traits>

#include "rg.pb.h"
#include "types/QueryIteratorImpl.h"
#include "types/SearchIteratorImpl.h"
#include "utils/Constants.h"
#include "utils/DmlUtils.h"
#include "utils/DqlUtils.h"
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
MilvusClientImpl::SetRpcDeadlineMs(uint64_t timeout_ms) {
    if (connection_ == nullptr) {
        return {StatusCode::NOT_CONNECTED, "Connection is not created!"};
    }
    connection_->GetConnectParam().SetRpcDeadlineMs(timeout_ms);
    return Status::OK();
}

Status
MilvusClientImpl::SetRetryParam(const RetryParam& retry_param) {
    if (connection_ == nullptr) {
        return {StatusCode::NOT_CONNECTED, "Connection is not created!"};
    }
    retry_param_ = retry_param;
    return Status::OK();
}

Status
MilvusClientImpl::GetVersion(std::string& version) {
    return GetServerVersion(version);
}

Status
MilvusClientImpl::GetServerVersion(std::string& version) {
    auto pre = []() {
        proto::milvus::GetVersionRequest rpc_request;
        return rpc_request;
    };

    auto post = [&version](const proto::milvus::GetVersionResponse& response) { version = response.version(); };

    return apiHandler<proto::milvus::GetVersionRequest, proto::milvus::GetVersionResponse>(
        pre, &MilvusConnection::GetVersion, post);
}

Status
MilvusClientImpl::GetSDKVersion(std::string& version) {
    version = GetBuildVersion();
    return Status::OK();
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
            // compile warning at this line since proto deprecates this method error_code()
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
                progress.total_ = 100;
                std::vector<std::string> partition_names;
                return getLoadingProgress(collection_name, partition_names, progress.finished_);
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
    return ListCollections(collections_info, false);
}

Status
MilvusClientImpl::ListCollections(CollectionsInfo& collections_info, bool only_show_loaded) {
    auto pre = [&only_show_loaded]() {
        proto::milvus::ShowCollectionsRequest rpc_request;
        auto show_type = only_show_loaded ? proto::milvus::ShowType::InMemory : proto::milvus::ShowType::All;
        rpc_request.set_type(show_type);
        return rpc_request;
    };

    auto post = [&collections_info](const proto::milvus::ShowCollectionsResponse& response) {
        collections_info.clear();
        for (int i = 0; i < response.collection_ids_size(); i++) {
            collections_info.emplace_back(response.collection_names(i), response.collection_ids(i),
                                          response.created_utc_timestamps(i));
        }
    };
    return apiHandler<proto::milvus::ShowCollectionsRequest, proto::milvus::ShowCollectionsResponse>(
        pre, &MilvusConnection::ShowCollections, post);
}

Status
MilvusClientImpl::GetLoadState(const std::string& collection_name, bool& is_loaded,
                               const std::vector<std::string> partition_names) {
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
        pre, &MilvusConnection::GetLoadState, post);
}

Status
MilvusClientImpl::AlterCollectionProperties(const std::string& collection_name,
                                            const std::unordered_map<std::string, std::string>& properties) {
    auto pre = [&collection_name, &properties]() {
        proto::milvus::AlterCollectionRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        for (const auto& pair : properties) {
            auto kv = rpc_request.add_properties();
            kv->set_key(pair.first);
            kv->set_value(pair.second);
        }
        return rpc_request;
    };

    return apiHandler<proto::milvus::AlterCollectionRequest, proto::common::Status>(pre,
                                                                                    &MilvusConnection::AlterCollection);
}

Status
MilvusClientImpl::DropCollectionProperties(const std::string& collection_name,
                                           const std::set<std::string>& property_keys) {
    auto pre = [&collection_name, &property_keys]() {
        proto::milvus::AlterCollectionRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        for (const auto& key : property_keys) {
            rpc_request.add_delete_keys(key);
        }
        return rpc_request;
    };

    return apiHandler<proto::milvus::AlterCollectionRequest, proto::common::Status>(pre,
                                                                                    &MilvusConnection::AlterCollection);
}

Status
MilvusClientImpl::AlterCollectionField(const std::string& collection_name, const std::string& field_name,
                                       const std::unordered_map<std::string, std::string>& properties) {
    auto pre = [&collection_name, &field_name, &properties]() {
        proto::milvus::AlterCollectionFieldRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_field_name(field_name);
        for (const auto& pair : properties) {
            auto kv = rpc_request.add_properties();
            kv->set_key(pair.first);
            kv->set_value(pair.second);
        }
        return rpc_request;
    };

    return apiHandler<proto::milvus::AlterCollectionFieldRequest, proto::common::Status>(
        pre, &MilvusConnection::AlterCollectionField);
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
                progress.total_ = 100;
                return getLoadingProgress(collection_name, partition_names, progress.finished_);
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
    return ListPartitions(collection_name, partitions_info, false);
}

Status
MilvusClientImpl::ListPartitions(const std::string& collection_name, PartitionsInfo& partitions_info,
                                 bool only_show_loaded) {
    auto pre = [&collection_name, &only_show_loaded] {
        proto::milvus::ShowPartitionsRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        auto show_type = only_show_loaded ? proto::milvus::ShowType::InMemory : proto::milvus::ShowType::All;
        rpc_request.set_type(show_type);  // compile warning at this line since proto deprecates this method set_type()
        return rpc_request;
    };

    auto post = [&partitions_info](const proto::milvus::ShowPartitionsResponse& response) {
        auto count = response.partition_names_size();
        partitions_info.clear();
        partitions_info.reserve(count);
        for (int i = 0; i < count; ++i) {
            partitions_info.emplace_back(response.partition_names(i), response.partitionids(i),
                                         response.created_timestamps(i));
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
MilvusClientImpl::DescribeAlias(const std::string& alias_name, AliasDesc& desc) {
    auto pre = [&alias_name]() {
        proto::milvus::DescribeAliasRequest rpc_request;
        rpc_request.set_alias(alias_name);
        return rpc_request;
    };

    auto post = [&desc](const proto::milvus::DescribeAliasResponse& response) {
        desc.SetName(response.alias());
        desc.SetDatabaseName(response.db_name());
        desc.SetCollectionName(response.collection());
    };

    return apiHandler<proto::milvus::DescribeAliasRequest, proto::milvus::DescribeAliasResponse>(
        pre, &MilvusConnection::DescribeAlias, post);
}

Status
MilvusClientImpl::ListAliases(const std::string& collection_name, std::vector<AliasDesc>& descs) {
    auto pre = [&collection_name]() {
        proto::milvus::ListAliasesRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        return rpc_request;
    };

    auto post = [&descs](const proto::milvus::ListAliasesResponse& response) {
        for (auto i = 0; i < response.aliases_size(); i++) {
            descs.emplace_back(response.aliases(i), response.db_name(), response.collection_name());
        }
    };

    return apiHandler<proto::milvus::ListAliasesRequest, proto::milvus::ListAliasesResponse>(
        pre, &MilvusConnection::ListAliases, post);
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
MilvusClientImpl::CurrentUsedDatabase(std::string& db_name) {
    if (connection_ == nullptr) {
        return {StatusCode::NOT_CONNECTED, "Connection is not created!"};
    }

    // the db name is returned from ConnectParam, the default db_name of ConnectParam
    // is an empty string which means the default database named "default".
    auto name = currentDbName("");
    db_name = name.empty() ? "default" : name;
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
        kv_pair->set_key(milvus::INDEX_TYPE);
        kv_pair->set_value(std::to_string(index_desc.IndexType()));

        // for scalar fields, no metric type
        if (index_desc.MetricType() != MetricType::DEFAULT) {
            kv_pair = rpc_request.add_extra_params();
            kv_pair->set_key(milvus::METRIC_TYPE);
            kv_pair->set_value(std::to_string(index_desc.MetricType()));
        }

        kv_pair = rpc_request.add_extra_params();
        kv_pair->set_key(milvus::PARAMS);
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
                if (key == milvus::INDEX_TYPE) {
                    index_desc.SetIndexType(IndexTypeCast(value));
                } else if (key == milvus::METRIC_TYPE) {
                    index_desc.SetMetricType(MetricTypeCast(value));
                } else if (key == milvus::PARAMS) {
                    index_desc.ExtraParamsFromJson(value);
                }
            }
        }
    };

    return apiHandler<proto::milvus::DescribeIndexRequest, proto::milvus::DescribeIndexResponse>(
        pre, &MilvusConnection::DescribeIndex, post);
}

Status
MilvusClientImpl::ListIndexes(const std::string& collection_name, const std::string& field_name,
                              std::vector<std::string>& index_names) {
    auto pre = [&collection_name, &field_name]() {
        proto::milvus::DescribeIndexRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_field_name(field_name);
        return rpc_request;
    };

    auto post = [&index_names, &field_name](const proto::milvus::DescribeIndexResponse& response) {
        auto count = response.index_descriptions_size();
        for (int i = 0; i < count; ++i) {
            auto rpc_desc = response.index_descriptions(i);
            if (field_name.empty() || field_name == rpc_desc.field_name()) {
                index_names.push_back(rpc_desc.index_name());
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
MilvusClientImpl::AlterIndexProperties(const std::string& collection_name, const std::string& index_name,
                                       const std::unordered_map<std::string, std::string>& properties) {
    auto pre = [&collection_name, &index_name, &properties]() {
        proto::milvus::AlterIndexRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_index_name(index_name);
        for (const auto& pair : properties) {
            auto kv_pair = rpc_request.add_extra_params();
            kv_pair->set_key(pair.first);
            kv_pair->set_value(pair.second);
        }

        return rpc_request;
    };

    return apiHandler<proto::milvus::AlterIndexRequest, proto::common::Status>(pre, &MilvusConnection::AlterIndex);
}

Status
MilvusClientImpl::DropIndexProperties(const std::string& collection_name, const std::string& index_name,
                                      const std::set<std::string>& property_keys) {
    auto pre = [&collection_name, &index_name, &property_keys]() {
        proto::milvus::AlterIndexRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_index_name(index_name);
        for (const auto& name : property_keys) {
            rpc_request.add_delete_keys(name);
        }

        return rpc_request;
    };

    return apiHandler<proto::milvus::AlterIndexRequest, proto::common::Status>(pre, &MilvusConnection::AlterIndex);
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
        enable_dynamic_field = collection_desc->Schema().EnableDynamicField();
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
            if (enable_dynamic_field && field->Name() == DYNAMIC_FIELD) {
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
        results.SetInsertCount(static_cast<uint64_t>(response.insert_cnt()));

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
MilvusClientImpl::Insert(const std::string& collection_name, const std::string& partition_name, const EntityRows& rows,
                         DmlResults& results) {
    std::vector<proto::schema::FieldData> rpc_fields;
    auto validate = [this, &collection_name, &rows, &rpc_fields]() {
        CollectionDescPtr collection_desc;
        auto status = getCollectionDesc(collection_name, false, collection_desc);
        if (!status.IsOk()) {
            return status;
        }
        return CheckAndSetRowData(rows, collection_desc->Schema(), false, rpc_fields);
    };

    auto pre = [&collection_name, &partition_name, &rows, &rpc_fields] {
        proto::milvus::InsertRequest rpc_request;

        auto* mutable_fields = rpc_request.mutable_fields_data();
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_partition_name(partition_name);
        rpc_request.set_num_rows(static_cast<uint32_t>(rows.size()));
        for (auto& field : rpc_fields) {
            mutable_fields->Add(std::move(field));
        }

        return rpc_request;
    };

    auto post = [this, &collection_name, &results](const proto::milvus::MutationResult& response) {
        auto id_array = CreateIDArray(response.ids());
        results.SetIdArray(std::move(id_array));
        results.SetTimestamp(response.timestamp());
        results.SetInsertCount(static_cast<uint64_t>(response.insert_cnt()));

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
        return Insert(collection_name, partition_name, rows, results);
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
        enable_dynamic_field = collection_desc->Schema().EnableDynamicField();
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
            if (enable_dynamic_field && field->Name() == DYNAMIC_FIELD) {
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
        results.SetUpsertCount(static_cast<uint64_t>(response.upsert_cnt()));

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
MilvusClientImpl::Upsert(const std::string& collection_name, const std::string& partition_name, const EntityRows& rows,
                         DmlResults& results) {
    std::vector<proto::schema::FieldData> rpc_fields;
    auto validate = [this, &collection_name, &rows, &rpc_fields]() {
        CollectionDescPtr collection_desc;
        auto status = getCollectionDesc(collection_name, false, collection_desc);
        if (!status.IsOk()) {
            return status;
        }
        return CheckAndSetRowData(rows, collection_desc->Schema(), true, rpc_fields);
    };

    auto pre = [&collection_name, &partition_name, &rows, &rpc_fields] {
        proto::milvus::UpsertRequest rpc_request;

        auto* mutable_fields = rpc_request.mutable_fields_data();
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_partition_name(partition_name);
        rpc_request.set_num_rows(static_cast<uint32_t>(rows.size()));
        for (auto& field : rpc_fields) {
            mutable_fields->Add(std::move(field));
        }
        return rpc_request;
    };

    auto post = [this, &collection_name, &results](const proto::milvus::MutationResult& response) {
        auto id_array = CreateIDArray(response.ids());
        results.SetIdArray(std::move(id_array));
        results.SetTimestamp(response.timestamp());
        results.SetUpsertCount(static_cast<uint64_t>(response.upsert_cnt()));

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
        return Upsert(collection_name, partition_name, rows, results);
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
        results.SetDeleteCount(static_cast<uint64_t>(response.delete_cnt()));

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
MilvusClientImpl::Search(const SearchArguments& arguments, SearchResults& results) {
    auto validate = [&arguments]() { return arguments.Validate(); };

    auto pre = [this, &arguments]() {
        proto::milvus::SearchRequest rpc_request;
        auto current_name = currentDbName(arguments.DatabaseName());
        ConvertSearchRequest(arguments, current_name, rpc_request);
        return rpc_request;
    };

    auto post = [this, &arguments, &results](const proto::milvus::SearchResults& response) {
        // in milvus version older than v2.4.20, the primary_field_name() is empty, we need to
        // get the primary key field name from collection schema
        const auto& result_data = response.results();
        auto pk_name = result_data.primary_field_name();
        if (result_data.primary_field_name().empty()) {
            CollectionDescPtr collection_desc;
            getCollectionDesc(arguments.CollectionName(), false, collection_desc);
            if (collection_desc != nullptr) {
                pk_name = collection_desc->Schema().Name();
            }
        }
        ConvertSearchResults(response, pk_name, results);
    };

    return apiHandler<proto::milvus::SearchRequest, proto::milvus::SearchResults>(
        validate, pre, &MilvusConnection::Search, nullptr, post);
}

Status
MilvusClientImpl::SearchIterator(SearchIteratorArguments& arguments, SearchIteratorPtr& iterator) {
    auto status = iteratorPrepare(arguments);
    if (!status.IsOk()) {
        return status;
    }

    // special process for search iterator
    // iterator needs vector field's metric type to determine the search range,
    // if user didn't offer the metric type, we need to describe the vector's index
    // to get the metric type.
    if (arguments.MetricType() == MetricType::DEFAULT) {
        std::string anns_field = arguments.AnnsField();
        if (anns_field.empty()) {
            CollectionDescPtr collection_desc;
            auto status = getCollectionDesc(arguments.CollectionName(), false, collection_desc);
            if (!status.IsOk()) {
                return status;
            }

            const auto& fields = collection_desc->Schema().Fields();
            std::set<std::string> vector_field_names;
            for (const auto& field : fields) {
                if (IsVectorType(field.FieldDataType())) {
                    vector_field_names.insert(field.Name());
                }
            }

            if (vector_field_names.empty()) {
                return {StatusCode::UNKNOWN_ERROR, "there should be at least one vector field in milvus collection"};
            }
            if (vector_field_names.size() > 1) {
                return {StatusCode::UNKNOWN_ERROR, "must specify anns_field when there are more than one vector field"};
            }
            anns_field = *(vector_field_names.begin());
        }

        IndexDesc desc;
        status = DescribeIndex(arguments.CollectionName(), anns_field, desc);
        if (!status.IsOk()) {
            return status;
        }
        arguments.SetMetricType(desc.MetricType());
    }

    // iterator constructor might throw exception when it fails to initialize
    try {
        iterator = std::make_shared<SearchIteratorImpl>(connection_, arguments, retry_param_);
    } catch (const std::exception& e) {
        return {StatusCode::UNKNOWN_ERROR, "Not able to create iterator, error: " + std::string(e.what())};
    }
    return Status::OK();
}

Status
MilvusClientImpl::HybridSearch(const HybridSearchArguments& arguments, SearchResults& results) {
    auto validate = [&arguments]() { return arguments.Validate(); };

    auto pre = [this, &arguments]() {
        proto::milvus::HybridSearchRequest rpc_request;
        auto current_name = currentDbName(arguments.DatabaseName());
        ConvertHybridSearchRequest(arguments, current_name, rpc_request);
        return rpc_request;
    };

    auto post = [this, &arguments, &results](const proto::milvus::SearchResults& response) {
        // in milvus version older than v2.4.20, the primary_field_name() is empty, we need to
        // get the primary key field name from collection schema
        const auto& result_data = response.results();
        auto pk_name = result_data.primary_field_name();
        if (result_data.primary_field_name().empty()) {
            CollectionDescPtr collection_desc;
            getCollectionDesc(arguments.CollectionName(), false, collection_desc);
            if (collection_desc != nullptr) {
                pk_name = collection_desc->Schema().Name();
            }
        }
        ConvertSearchResults(response, pk_name, results);
    };

    return apiHandler<proto::milvus::HybridSearchRequest, proto::milvus::SearchResults>(
        validate, pre, &MilvusConnection::HybridSearch, nullptr, post);
}

Status
MilvusClientImpl::Query(const QueryArguments& arguments, QueryResults& results) {
    auto pre = [this, &arguments]() {
        proto::milvus::QueryRequest rpc_request;
        auto current_name = currentDbName(arguments.DatabaseName());
        ConvertQueryRequest(arguments, current_name, rpc_request);
        return rpc_request;
    };

    auto post = [&results](const proto::milvus::QueryResults& response) { ConvertQueryResults(response, results); };
    return apiHandler<proto::milvus::QueryRequest, proto::milvus::QueryResults>(pre, &MilvusConnection::Query, post);
}

Status
MilvusClientImpl::QueryIterator(QueryIteratorArguments& arguments, QueryIteratorPtr& iterator) {
    auto status = iteratorPrepare(arguments);
    if (!status.IsOk()) {
        return status;
    }

    // iterator constructor might throw exception when it fails to initialize
    try {
        iterator = std::make_shared<QueryIteratorImpl>(connection_, arguments, retry_param_);
    } catch (const std::exception& e) {
        return {StatusCode::UNKNOWN_ERROR, "Not able to create iterator, error: " + std::string(e.what())};
    }
    return Status::OK();
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
            std::vector<int64_t> ids;
            for (auto id : info.nodeids()) {
                ids.push_back(id);
            }
            segments_info.emplace_back(info.collectionid(), info.partitionid(), info.segmentid(), info.num_rows(),
                                       milvus::SegmentStateCast(info.state()), info.index_name(), info.indexid(), ids);
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
    return CreateCredential(username, password);
}

Status
MilvusClientImpl::UpdateCredential(const std::string& username, const std::string& old_password,
                                   const std::string& new_password) {
    return UpdatePassword(username, old_password, new_password);
}

Status
MilvusClientImpl::DeleteCredential(const std::string& username) {
    return DropUser(username);
}

Status
MilvusClientImpl::ListCredUsers(std::vector<std::string>& users) {
    return ListUsers(users);
}

Status
MilvusClientImpl::CreateResourceGroup(const std::string& name, const ResourceGroupConfig& config) {
    auto pre = [&name, &config]() {
        proto::milvus::CreateResourceGroupRequest rpc_request;
        rpc_request.set_resource_group(name);

        auto rpc_config = new proto::rg::ResourceGroupConfig{};
        ConvertResourceGroupConfig(config, rpc_config);
        rpc_request.set_allocated_config(rpc_config);
        return rpc_request;
    };

    return apiHandler<proto::milvus::CreateResourceGroupRequest, proto::common::Status>(
        pre, &MilvusConnection::CreateResourceGroup, nullptr);
}

Status
MilvusClientImpl::DropResourceGroup(const std::string& name) {
    auto pre = [&name]() {
        proto::milvus::DropResourceGroupRequest rpc_request;
        rpc_request.set_resource_group(name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::DropResourceGroupRequest, proto::common::Status>(
        pre, &MilvusConnection::DropResourceGroup, nullptr);
}

Status
MilvusClientImpl::UpdateResourceGroups(const std::unordered_map<std::string, ResourceGroupConfig>& groups) {
    auto pre = [&groups]() {
        proto::milvus::UpdateResourceGroupsRequest rpc_request;
        for (const auto& pair : groups) {
            proto::rg::ResourceGroupConfig rpc_config;
            ConvertResourceGroupConfig(pair.second, &rpc_config);
            rpc_request.mutable_resource_groups()->insert(std::make_pair(pair.first, rpc_config));
        }
        return rpc_request;
    };

    return apiHandler<proto::milvus::UpdateResourceGroupsRequest, proto::common::Status>(
        pre, &MilvusConnection::UpdateResourceGroups, nullptr);
}

Status
MilvusClientImpl::TransferNode(const std::string& source_group, const std::string& target_group, uint32_t num_nodes) {
    auto pre = [&source_group, &target_group, &num_nodes]() {
        proto::milvus::TransferNodeRequest rpc_request;
        rpc_request.set_source_resource_group(source_group);
        rpc_request.set_target_resource_group(target_group);
        rpc_request.set_num_node(num_nodes);
        return rpc_request;
    };

    return apiHandler<proto::milvus::TransferNodeRequest, proto::common::Status>(pre, &MilvusConnection::TransferNode,
                                                                                 nullptr);
}

Status
MilvusClientImpl::TransferReplica(const std::string& source_group, const std::string& target_group,
                                  const std::string& collection_name, uint32_t num_replicas) {
    auto pre = [&source_group, &target_group, &collection_name, &num_replicas]() {
        proto::milvus::TransferReplicaRequest rpc_request;
        rpc_request.set_source_resource_group(source_group);
        rpc_request.set_target_resource_group(target_group);
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_num_replica(num_replicas);
        return rpc_request;
    };

    return apiHandler<proto::milvus::TransferReplicaRequest, proto::common::Status>(
        pre, &MilvusConnection::TransferReplica, nullptr);
}

Status
MilvusClientImpl::ListResourceGroups(std::vector<std::string>& group_names) {
    auto pre = []() {
        proto::milvus::ListResourceGroupsRequest rpc_request;
        return rpc_request;
    };

    auto post = [&group_names](const proto::milvus::ListResourceGroupsResponse& response) {
        group_names.clear();
        for (const auto& group : response.resource_groups()) {
            group_names.push_back(group);
        }
    };
    return apiHandler<proto::milvus::ListResourceGroupsRequest, proto::milvus::ListResourceGroupsResponse>(
        pre, &MilvusConnection::ListResourceGroups, post);
}

Status
MilvusClientImpl::DescribeResourceGroup(const std::string& group_name, ResourceGroupDesc& desc) {
    auto pre = [&group_name]() {
        proto::milvus::DescribeResourceGroupRequest rpc_request;
        rpc_request.set_resource_group(group_name);
        return rpc_request;
    };

    auto post = [&desc](const proto::milvus::DescribeResourceGroupResponse& response) {
        const auto& group = response.resource_group();
        desc.SetName(group.name());
        desc.SetCapacity(static_cast<uint32_t>(group.capacity()));
        desc.SetAvailableNodesNum(static_cast<uint32_t>(group.num_available_node()));

        for (const auto& pair : group.num_loaded_replica()) {
            desc.AddLoadedReplicasNum(pair.first, static_cast<uint32_t>(pair.second));
        }
        for (const auto& pair : group.num_outgoing_node()) {
            desc.AddOutgoingNodesNum(pair.first, static_cast<uint32_t>(pair.second));
        }
        for (const auto& pair : group.num_incoming_node()) {
            desc.AddIncomingNodesNum(pair.first, static_cast<uint32_t>(pair.second));
        }

        ResourceGroupConfig config;
        ConvertResourceGroupConfig(group.config(), config);
        desc.SetConfig(std::move(config));

        for (const auto& info : group.nodes()) {
            desc.AddNode({info.node_id(), info.address(), info.hostname()});
        }
    };
    return apiHandler<proto::milvus::DescribeResourceGroupRequest, proto::milvus::DescribeResourceGroupResponse>(
        pre, &MilvusConnection::DescribeResourceGroup, post);
}

Status
MilvusClientImpl::CreateUser(const std::string& user_name, const std::string& password) {
    auto pre = [&user_name, &password]() {
        proto::milvus::CreateCredentialRequest rpc_request;
        rpc_request.set_username(user_name);
        rpc_request.set_password(milvus::Base64Encode(password));
        return rpc_request;
    };

    return apiHandler<proto::milvus::CreateCredentialRequest, proto::common::Status>(
        pre, &MilvusConnection::CreateCredential, nullptr);
}

Status
MilvusClientImpl::UpdatePassword(const std::string& user_name, const std::string& old_password,
                                 const std::string& new_password) {
    auto pre = [&user_name, &old_password, &new_password]() {
        proto::milvus::UpdateCredentialRequest rpc_request;
        rpc_request.set_username(user_name);
        rpc_request.set_oldpassword(milvus::Base64Encode(old_password));
        rpc_request.set_newpassword(milvus::Base64Encode(new_password));
        return rpc_request;
    };

    return apiHandler<proto::milvus::UpdateCredentialRequest, proto::common::Status>(
        pre, &MilvusConnection::UpdateCredential, nullptr);
}

Status
MilvusClientImpl::DropUser(const std::string& user_name) {
    auto pre = [&user_name]() {
        proto::milvus::DeleteCredentialRequest rpc_request;
        rpc_request.set_username(user_name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::DeleteCredentialRequest, proto::common::Status>(
        pre, &MilvusConnection::DeleteCredential, nullptr);
}

Status
MilvusClientImpl::DescribeUser(const std::string& user_name, UserDesc& desc) {
    auto pre = [&user_name]() {
        proto::milvus::SelectUserRequest rpc_request;
        rpc_request.mutable_user()->set_name(user_name);
        rpc_request.set_include_role_info(true);
        return rpc_request;
    };

    auto post = [&user_name, &desc](const proto::milvus::SelectUserResponse& response) {
        desc.SetName(user_name);
        if (response.results().size() > 0) {
            auto result = response.results().at(0);
            for (const auto& role : result.roles()) {
                desc.AddRole(role.name());
            }
        }
    };

    return apiHandler<proto::milvus::SelectUserRequest, proto::milvus::SelectUserResponse>(
        pre, &MilvusConnection::SelectUser, post);
}

Status
MilvusClientImpl::ListUsers(std::vector<std::string>& names) {
    auto pre = []() {
        proto::milvus::ListCredUsersRequest rpc_request;
        return rpc_request;
    };

    auto post = [&names](const proto::milvus::ListCredUsersResponse& response) {
        names.clear();
        for (const auto& user : response.usernames()) {
            names.emplace_back(user);
        }
    };

    return apiHandler<proto::milvus::ListCredUsersRequest, proto::milvus::ListCredUsersResponse>(
        pre, &MilvusConnection::ListCredUsers, post);
}

Status
MilvusClientImpl::CreateRole(const std::string& role_name) {
    auto pre = [&role_name]() {
        proto::milvus::CreateRoleRequest rpc_request;
        rpc_request.mutable_entity()->set_name(role_name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::CreateRoleRequest, proto::common::Status>(pre, &MilvusConnection::CreateRole,
                                                                               nullptr);
}

Status
MilvusClientImpl::DropRole(const std::string& role_name, bool force_drop) {
    auto pre = [&role_name, &force_drop]() {
        proto::milvus::DropRoleRequest rpc_request;
        rpc_request.set_role_name(role_name);
        rpc_request.set_force_drop(force_drop);
        return rpc_request;
    };

    return apiHandler<proto::milvus::DropRoleRequest, proto::common::Status>(pre, &MilvusConnection::DropRole, nullptr);
}

Status
MilvusClientImpl::DescribeRole(const std::string& role_name, RoleDesc& desc) {
    auto pre = [&role_name]() {
        proto::milvus::SelectGrantRequest rpc_request;
        rpc_request.mutable_entity()->mutable_role()->set_name(role_name);
        return rpc_request;
    };

    auto post = [&role_name, &desc](const proto::milvus::SelectGrantResponse& response) {
        desc.SetName(role_name);
        for (const auto& entity : response.entities()) {
            desc.AddGrantItem({entity.object().name(), entity.object_name(), entity.db_name(), entity.role().name(),
                               entity.grantor().user().name(), entity.grantor().privilege().name()});
        }
    };

    return apiHandler<proto::milvus::SelectGrantRequest, proto::milvus::SelectGrantResponse>(
        pre, &MilvusConnection::SelectGrant, post);
}

Status
MilvusClientImpl::ListRoles(std::vector<std::string>& names) {
    auto pre = []() {
        proto::milvus::SelectRoleRequest rpc_request;
        rpc_request.set_include_user_info(false);
        return rpc_request;
    };

    auto post = [&names](const proto::milvus::SelectRoleResponse& response) {
        names.clear();
        for (const auto& result : response.results()) {
            names.push_back(result.role().name());
        }
    };

    return apiHandler<proto::milvus::SelectRoleRequest, proto::milvus::SelectRoleResponse>(
        pre, &MilvusConnection::SelectRole, post);
}

Status
MilvusClientImpl::GrantRole(const std::string& user_name, const std::string& role_name) {
    auto pre = [&user_name, &role_name]() {
        proto::milvus::OperateUserRoleRequest rpc_request;
        rpc_request.set_username(user_name);
        rpc_request.set_role_name(role_name);
        rpc_request.set_type(proto::milvus::OperateUserRoleType::AddUserToRole);
        return rpc_request;
    };

    return apiHandler<proto::milvus::OperateUserRoleRequest, proto::common::Status>(
        pre, &MilvusConnection::OperateUserRole, nullptr);
}

Status
MilvusClientImpl::RevokeRole(const std::string& user_name, const std::string& role_name) {
    auto pre = [&user_name, &role_name]() {
        proto::milvus::OperateUserRoleRequest rpc_request;
        rpc_request.set_username(user_name);
        rpc_request.set_role_name(role_name);
        rpc_request.set_type(proto::milvus::OperateUserRoleType::RemoveUserFromRole);
        return rpc_request;
    };

    return apiHandler<proto::milvus::OperateUserRoleRequest, proto::common::Status>(
        pre, &MilvusConnection::OperateUserRole, nullptr);
}

Status
MilvusClientImpl::GrantPrivilege(const std::string& role_name, const std::string& privilege,
                                 const std::string& collection_name, const std::string& db_name) {
    auto pre = [&role_name, &privilege, &collection_name, &db_name]() {
        proto::milvus::OperatePrivilegeV2Request rpc_request;
        rpc_request.mutable_role()->set_name(role_name);
        rpc_request.mutable_grantor()->mutable_privilege()->set_name(privilege);
        rpc_request.set_type(proto::milvus::OperatePrivilegeType::Grant);
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_db_name(db_name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::OperatePrivilegeV2Request, proto::common::Status>(
        pre, &MilvusConnection::OperatePrivilegeV2, nullptr);
}

Status
MilvusClientImpl::RevokePrivilege(const std::string& role_name, const std::string& privilege,
                                  const std::string& collection_name, const std::string& db_name) {
    auto pre = [&role_name, &privilege, &collection_name, &db_name]() {
        proto::milvus::OperatePrivilegeV2Request rpc_request;
        rpc_request.mutable_role()->set_name(role_name);
        rpc_request.mutable_grantor()->mutable_privilege()->set_name(privilege);
        rpc_request.set_type(proto::milvus::OperatePrivilegeType::Revoke);
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_db_name(db_name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::OperatePrivilegeV2Request, proto::common::Status>(
        pre, &MilvusConnection::OperatePrivilegeV2, nullptr);
}

Status
MilvusClientImpl::CreatePrivilegeGroup(const std::string& group_name) {
    auto pre = [&group_name]() {
        proto::milvus::CreatePrivilegeGroupRequest rpc_request;
        rpc_request.set_group_name(group_name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::CreatePrivilegeGroupRequest, proto::common::Status>(
        pre, &MilvusConnection::CreatePrivilegeGroup, nullptr);
}

Status
MilvusClientImpl::DropPrivilegeGroup(const std::string& group_name) {
    auto pre = [&group_name]() {
        proto::milvus::DropPrivilegeGroupRequest rpc_request;
        rpc_request.set_group_name(group_name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::DropPrivilegeGroupRequest, proto::common::Status>(
        pre, &MilvusConnection::DropPrivilegeGroup, nullptr);
}

Status
MilvusClientImpl::ListPrivilegeGroups(PrivilegeGroupInfos& groups) {
    auto pre = []() {
        proto::milvus::ListPrivilegeGroupsRequest rpc_request;
        return rpc_request;
    };

    auto post = [&groups](const proto::milvus::ListPrivilegeGroupsResponse& response) {
        groups.clear();
        for (const auto& result : response.privilege_groups()) {
            std::vector<std::string> privileges;
            for (const auto& rpc_privilege : result.privileges()) {
                privileges.push_back(rpc_privilege.name());
            }
            groups.emplace_back(result.group_name(), std::move(privileges));
        }
    };

    return apiHandler<proto::milvus::ListPrivilegeGroupsRequest, proto::milvus::ListPrivilegeGroupsResponse>(
        pre, &MilvusConnection::ListPrivilegeGroups, post);
}

Status
MilvusClientImpl::AddPrivilegesToGroup(const std::string& group_name, const std::vector<std::string>& privileges) {
    auto pre = [&group_name, &privileges]() {
        proto::milvus::OperatePrivilegeGroupRequest rpc_request;
        rpc_request.set_group_name(group_name);
        for (const auto& privilege : privileges) {
            auto rpc_privilege = rpc_request.mutable_privileges()->Add();
            rpc_privilege->set_name(privilege);
        }
        rpc_request.set_type(proto::milvus::OperatePrivilegeGroupType::AddPrivilegesToGroup);
        return rpc_request;
    };

    return apiHandler<proto::milvus::OperatePrivilegeGroupRequest, proto::common::Status>(
        pre, &MilvusConnection::OperatePrivilegeGroup, nullptr);
}

Status
MilvusClientImpl::RemovePrivilegesFromGroup(const std::string& group_name, const std::vector<std::string>& privileges) {
    auto pre = [&group_name, &privileges]() {
        proto::milvus::OperatePrivilegeGroupRequest rpc_request;
        rpc_request.set_group_name(group_name);
        for (const auto& privilege : privileges) {
            auto rpc_privilege = rpc_request.mutable_privileges()->Add();
            rpc_privilege->set_name(privilege);
        }
        rpc_request.set_type(proto::milvus::OperatePrivilegeGroupType::RemovePrivilegesFromGroup);
        return rpc_request;
    };

    return apiHandler<proto::milvus::OperatePrivilegeGroupRequest, proto::common::Status>(
        pre, &MilvusConnection::OperatePrivilegeGroup, nullptr);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// internal used methods
Status
MilvusClientImpl::getLoadingProgress(const std::string& collection_name, const std::vector<std::string> partition_names,
                                     uint32_t& progress) {
    proto::milvus::GetLoadingProgressRequest progress_req;
    progress_req.set_collection_name(collection_name);
    for (const auto& partition_name : partition_names) {
        progress_req.add_partition_names(partition_name);
    }
    proto::milvus::GetLoadingProgressResponse progress_resp;
    uint64_t timeout = connection_->GetConnectParam().RpcDeadlineMs();
    auto status = connection_->GetLoadingProgress(progress_req, progress_resp, GrpcOpts{timeout});
    if (!status.IsOk()) {
        return status;
    }
    progress = static_cast<uint32_t>(progress_resp.progress());
    return Status::OK();
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

template <typename ArgClass>
Status
MilvusClientImpl::iteratorPrepare(ArgClass& arguments) {
    CollectionDescPtr collection_desc;
    auto status = getCollectionDesc(arguments.CollectionName(), false, collection_desc);
    if (!status.IsOk()) {
        return status;
    }
    arguments.SetCollectionID(collection_desc->ID());

    const auto& fields = collection_desc->Schema().Fields();
    bool pk_found = false;
    for (const auto& field : fields) {
        if (field.IsPrimaryKey()) {
            arguments.SetPkSchema(field);
            pk_found = true;
            break;
        }
    }
    if (!pk_found) {
        return {StatusCode::UNKNOWN_ERROR, "Primary key field is not found"};
    }
    return Status::OK();
}

}  // namespace milvus
