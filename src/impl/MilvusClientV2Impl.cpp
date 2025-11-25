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

#include "MilvusClientV2Impl.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include <type_traits>

#include "rg.pb.h"
#include "types/QueryIteratorImpl.h"
#include "types/SearchIteratorImpl.h"
#include "types/SearchIteratorV2Impl.h"
#include "utils/Constants.h"
#include "utils/DmlUtils.h"
#include "utils/DqlUtils.h"
#include "utils/FieldDataSchema.h"
#include "utils/GtsDict.h"
#include "utils/TypeUtils.h"

namespace milvus {

std::shared_ptr<MilvusClientV2Impl>
MilvusClientV2Impl::Create() {
    return std::make_shared<MilvusClientV2Impl>();
}

MilvusClientV2Impl::~MilvusClientV2Impl() {
    Disconnect();
}

Status
MilvusClientV2Impl::Connect(const ConnectParam& param) {
    return connection_.Connect(param);
}

Status
MilvusClientV2Impl::Disconnect() {
    return connection_.Disconnect();
}

Status
MilvusClientV2Impl::SetRpcDeadlineMs(uint64_t timeout_ms) {
    return connection_.SetRpcDeadlineMs(timeout_ms);
}

Status
MilvusClientV2Impl::SetRetryParam(const RetryParam& retry_param) {
    return connection_.SetRetryParam(retry_param);
}

Status
MilvusClientV2Impl::GetServerVersion(std::string& version) {
    auto post = [&version](const proto::milvus::GetVersionResponse& response) {
        version = response.version();
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::GetVersionRequest, proto::milvus::GetVersionResponse>(
        nullptr, &MilvusConnection::GetVersion, post);
}

Status
MilvusClientV2Impl::GetSDKVersion(std::string& version) {
    version = GetBuildVersion();
    return Status::OK();
}

Status
MilvusClientV2Impl::CheckHealth(const CheckHealthRequest& request, CheckHealthResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::CreateCollection(const CreateCollectionRequest& request) {
    const auto& schemaPtr = request.CollectionSchema();
    if (schemaPtr == nullptr) {
        return {StatusCode::INVALID_AGUMENT, "Collection schema is null"};
    }

    CollectionSchema& schema = *schemaPtr;
    auto validate = [&schema]() {
        for (const auto& field : schema.Fields()) {
            auto status = CheckDefaultValue(field);
            if (!status.IsOk()) {
                return status;
            }
        }
        return Status::OK();
    };

    auto pre = [&schema, &request](proto::milvus::CreateCollectionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(schema.Name());
        rpc_request.set_shards_num(schema.ShardsNum());
        rpc_request.set_consistency_level(ConsistencyLevelCast(request.GetConsistencyLevel()));
        if (request.NumPartitions() > 0) {
            rpc_request.set_num_partitions(request.NumPartitions());
        }

        // properties
        for (auto it : request.Properties()) {
            auto kv = rpc_request.add_properties();
            kv->set_key(it.first);
            kv->set_value(it.second);
        }

        // schema
        proto::schema::CollectionSchema rpc_collection;
        ConvertCollectionSchema(schema, rpc_collection);

        std::string binary;
        rpc_collection.SerializeToString(&binary);
        rpc_request.set_schema(binary);
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::CreateCollectionRequest, proto::common::Status>(
        validate, pre, &MilvusConnection::CreateCollection, nullptr);
}

Status
MilvusClientV2Impl::HasCollection(const HasCollectionRequest& request, HasCollectionResponse& response) {
    auto pre = [&request](proto::milvus::HasCollectionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_time_stamp(0);
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::BoolResponse& rpc_response) {
        response.SetHas(rpc_response.value());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::HasCollectionRequest, proto::milvus::BoolResponse>(
        pre, &MilvusConnection::HasCollection, post);
}

Status
MilvusClientV2Impl::DropCollection(const DropCollectionRequest& request) {
    auto pre = [&request](proto::milvus::DropCollectionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        return Status::OK();
    };

    auto post = [this, &request](const proto::common::Status& status) {
        if (status.error_code() == proto::common::ErrorCode::Success && status.code() == 0) {
            // compile warning at this line since proto deprecates this method error_code()
            // TODO: if the parameters provides db_name in future, we need to set the correct
            // db_name to RemoveCollectionTs()
            auto db_name = connection_.CurrentDbName(request.DatabaseName());
            auto collection_name = request.CollectionName();
            GtsDict::GetInstance().RemoveCollectionTs(connection_.CurrentDbName(db_name), collection_name);
            removeCollectionDesc(db_name, collection_name);
        }
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::DropCollectionRequest, proto::common::Status>(
        pre, &MilvusConnection::DropCollection, post);
}

Status
MilvusClientV2Impl::LoadCollection(const LoadCollectionRequest& request) {
    auto pre = [&request](proto::milvus::LoadCollectionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_replica_number(request.ReplicaNum());
        rpc_request.set_skip_load_dynamic_field(request.SkipDynamicField());
        rpc_request.set_refresh(request.Refresh());
        for (const auto& fn : request.LoadFields()) {
            rpc_request.add_load_fields(fn);
        }
        for (const auto& rg : request.TargetResourceGroups()) {
            rpc_request.add_resource_groups(rg);
        }
        return Status::OK();
    };

    // if not sync mode, directly return
    if (!request.Sync()) {
        return connection_.Invoke<proto::milvus::LoadCollectionRequest, proto::common::Status>(
            pre, &MilvusConnection::LoadCollection);
    }

    // TODO: check timeout value in sync mode
    ProgressMonitor progress_monitor = ProgressMonitor::Forever();
    auto wait_for_status = [this, &request, &progress_monitor](const proto::common::Status&) {
        return ConnectionHandler::WaitForStatus(
            [&request, this](Progress& progress) -> Status {
                progress.total_ = 100;
                auto db_name = connection_.CurrentDbName(request.DatabaseName());
                std::vector<std::string> partition_names;
                return connection_.GetLoadingProgress(db_name, request.CollectionName(), partition_names,
                                                      progress.finished_);
            },
            progress_monitor);
    };
    return connection_.Invoke<proto::milvus::LoadCollectionRequest, proto::common::Status>(
        pre, &MilvusConnection::LoadCollection, wait_for_status);
}

Status
MilvusClientV2Impl::ReleaseCollection(const ReleaseCollectionRequest& request) {
    auto pre = [&request](proto::milvus::ReleaseCollectionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::ReleaseCollectionRequest, proto::common::Status>(
        pre, &MilvusConnection::ReleaseCollection);
}

Status
MilvusClientV2Impl::DescribeCollection(const DescribeCollectionRequest& request, DescribeCollectionResponse& response) {
    auto pre = [&request](proto::milvus::DescribeCollectionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::DescribeCollectionResponse& rpc_response) {
        CollectionSchema schema;
        ConvertCollectionSchema(rpc_response.schema(), schema);
        schema.SetShardsNum(rpc_response.shards_num());

        CollectionDesc collection_desc;
        collection_desc.SetSchema(std::move(schema));
        collection_desc.SetID(rpc_response.collectionid());
        collection_desc.SetCreatedTime(rpc_response.created_timestamp());

        std::vector<std::string> aliases;
        aliases.reserve(rpc_response.aliases_size());
        aliases.insert(aliases.end(), rpc_response.aliases().begin(), rpc_response.aliases().end());
        collection_desc.SetAlias(std::move(aliases));

        response.SetDesc(std::move(collection_desc));

        // TODO: set properties
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::DescribeCollectionRequest, proto::milvus::DescribeCollectionResponse>(
        pre, &MilvusConnection::DescribeCollection, post);
}

Status
MilvusClientV2Impl::RenameCollection(const RenameCollectionRequest& request) {
    auto pre = [&request](proto::milvus::RenameCollectionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_oldname(request.CollectionName());
        rpc_request.set_newname(request.NewCollectionName());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::RenameCollectionRequest, proto::common::Status>(
        pre, &MilvusConnection::RenameCollection);
}

Status
MilvusClientV2Impl::GetCollectionStats(const GetCollectionStatsRequest& request, GetCollectionStatsResponse& response) {
    auto pre = [&request](proto::milvus::GetCollectionStatisticsRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        return Status::OK();
    };

    auto post = [&request, &response](const proto::milvus::GetCollectionStatisticsResponse& rpc_response) {
        CollectionStat collection_stat;
        collection_stat.SetName(request.CollectionName());
        for (const auto& stat_pair : rpc_response.stats()) {
            collection_stat.Emplace(stat_pair.key(), stat_pair.value());
        }
        response.SetStats(std::move(collection_stat));
        return Status::OK();
    };

    return connection_
        .Invoke<proto::milvus::GetCollectionStatisticsRequest, proto::milvus::GetCollectionStatisticsResponse>(
            pre, &MilvusConnection::GetCollectionStatistics, post);
}

Status
MilvusClientV2Impl::ListCollections(const ListCollectionsRequest& request, ListCollectionsResponse& response) {
    auto pre = [&request](proto::milvus::ShowCollectionsRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        auto show_type = request.OnlyShowLoaded() ? proto::milvus::ShowType::InMemory : proto::milvus::ShowType::All;
        rpc_request.set_type(show_type);
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::ShowCollectionsResponse& rpc_response) {
        std::vector<std::string> collection_names;
        std::vector<CollectionInfo> collection_infos;
        for (int i = 0; i < rpc_response.collection_ids_size(); i++) {
            collection_names.push_back(rpc_response.collection_names(i));
            collection_infos.emplace_back(rpc_response.collection_names(i), rpc_response.collection_ids(i),
                                          rpc_response.created_utc_timestamps(i));
        }
        response.SetCollectionNames(std::move(collection_names));
        response.SetCollectionInfos(std::move(collection_infos));
        return Status::OK();
    };
    return connection_.Invoke<proto::milvus::ShowCollectionsRequest, proto::milvus::ShowCollectionsResponse>(
        pre, &MilvusConnection::ShowCollections, post);
}

Status
MilvusClientV2Impl::GetLoadState(const GetLoadStateRequest& request, GetLoadStateResponse& response) {
    auto pre = [&request](proto::milvus::GetLoadStateRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        for (const auto& partition_name : request.PartitionNames()) {
            rpc_request.add_partition_names(partition_name);
        }
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::GetLoadStateResponse& rpc_response) {
        response.SetState(LoadStateCast(rpc_response.state()));

        // TODO: set progress percent if state is LoadStateLoading
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::GetLoadStateRequest, proto::milvus::GetLoadStateResponse>(
        pre, &MilvusConnection::GetLoadState, post);
}

Status
MilvusClientV2Impl::AlterCollectionProperties(const AlterCollectionPropertiesRequest& request) {
    auto pre = [&request](proto::milvus::AlterCollectionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        for (const auto& pair : request.Properties()) {
            auto kv = rpc_request.add_properties();
            kv->set_key(pair.first);
            kv->set_value(pair.second);
        }
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::AlterCollectionRequest, proto::common::Status>(
        pre, &MilvusConnection::AlterCollection);
}

Status
MilvusClientV2Impl::DropCollectionProperties(const DropCollectionPropertiesRequest& request) {
    auto pre = [&request](proto::milvus::AlterCollectionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        for (const auto& key : request.PropertyKeys()) {
            rpc_request.add_delete_keys(key);
        }
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::AlterCollectionRequest, proto::common::Status>(
        pre, &MilvusConnection::AlterCollection);
}

Status
MilvusClientV2Impl::AlterCollectionFieldProperties(const AlterCollectionFieldPropertiesRequest& request) {
    auto pre = [&request](proto::milvus::AlterCollectionFieldRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_field_name(request.FieldName());
        for (const auto& pair : request.Properties()) {
            auto kv = rpc_request.add_properties();
            kv->set_key(pair.first);
            kv->set_value(pair.second);
        }
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::AlterCollectionFieldRequest, proto::common::Status>(
        pre, &MilvusConnection::AlterCollectionField);
}

Status
MilvusClientV2Impl::DropCollectionFieldProperties(const DropCollectionFieldPropertiesRequest& request) {
    auto pre = [&request](proto::milvus::AlterCollectionFieldRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_field_name(request.FieldName());
        for (const auto& key : request.PropertyKeys()) {
            rpc_request.add_delete_keys(key);
        }
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::AlterCollectionFieldRequest, proto::common::Status>(
        pre, &MilvusConnection::AlterCollectionField);
}

Status
MilvusClientV2Impl::CreatePartition(const CreatePartitionRequest& request) {
    auto pre = [&request](proto::milvus::CreatePartitionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_partition_name(request.PartitionName());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::CreatePartitionRequest, proto::common::Status>(
        pre, &MilvusConnection::CreatePartition);
}

Status
MilvusClientV2Impl::DropPartition(const DropPartitionRequest& request) {
    auto pre = [&request](proto::milvus::DropPartitionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_partition_name(request.PartitionName());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::DropPartitionRequest, proto::common::Status>(
        pre, &MilvusConnection::DropPartition);
}

Status
MilvusClientV2Impl::HasPartition(const HasPartitionRequest& request, HasPartitionResponse& response) {
    auto pre = [&request](proto::milvus::HasPartitionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_partition_name(request.PartitionName());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::BoolResponse& rpc_response) {
        response.SetHas(rpc_response.value());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::HasPartitionRequest, proto::milvus::BoolResponse>(
        pre, &MilvusConnection::HasPartition, post);
}

Status
MilvusClientV2Impl::LoadPartitions(const LoadPartitionsRequest& request) {
    auto pre = [&request](proto::milvus::LoadPartitionsRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_replica_number(request.ReplicaNum());
        rpc_request.set_skip_load_dynamic_field(request.SkipDynamicField());
        rpc_request.set_refresh(request.Refresh());
        for (const auto& partition_name : request.PartitionNames()) {
            rpc_request.add_partition_names(partition_name);
        }
        for (const auto& fn : request.LoadFields()) {
            rpc_request.add_load_fields(fn);
        }
        for (const auto& rg : request.TargetResourceGroups()) {
            rpc_request.add_resource_groups(rg);
        }
        return Status::OK();
    };

    // if not sync mode, directly return
    if (!request.Sync()) {
        return connection_.Invoke<proto::milvus::LoadPartitionsRequest, proto::common::Status>(
            pre, &MilvusConnection::LoadPartitions);
    }

    // TODO: check timeout value in sync mode
    ProgressMonitor progress_monitor = ProgressMonitor::Forever();
    auto wait_for_status = [this, &request, &progress_monitor](const proto::common::Status&) {
        return ConnectionHandler::WaitForStatus(
            [&request, this](Progress& progress) -> Status {
                progress.total_ = 100;
                auto db_name = connection_.CurrentDbName(request.DatabaseName());
                return connection_.GetLoadingProgress(db_name, request.CollectionName(), request.PartitionNames(),
                                                      progress.finished_);
            },
            progress_monitor);
    };
    return connection_.Invoke<proto::milvus::LoadPartitionsRequest, proto::common::Status>(
        nullptr, pre, &MilvusConnection::LoadPartitions, wait_for_status, nullptr);
}

Status
MilvusClientV2Impl::ReleasePartitions(const ReleasePartitionsRequest& request) {
    auto pre = [&request](proto::milvus::ReleasePartitionsRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        for (const auto& partition_name : request.PartitionNames()) {
            rpc_request.add_partition_names(partition_name);
        }
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::ReleasePartitionsRequest, proto::common::Status>(
        pre, &MilvusConnection::ReleasePartitions);
}

Status
MilvusClientV2Impl::GetPartitionStatistics(const GetPartitionStatsRequest& request,
                                           GetPartitionStatsResponse& response) {
    auto pre = [&request](proto::milvus::GetPartitionStatisticsRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_partition_name(request.PartitionName());
        return Status::OK();
    };

    auto post = [&request, &response](const proto::milvus::GetPartitionStatisticsResponse& rpc_response) {
        PartitionStat partition_stat;
        partition_stat.SetName(request.PartitionName());
        for (const auto& stat_pair : rpc_response.stats()) {
            partition_stat.Emplace(stat_pair.key(), stat_pair.value());
        }

        response.SetStats(std::move(partition_stat));
        return Status::OK();
    };

    return connection_
        .Invoke<proto::milvus::GetPartitionStatisticsRequest, proto::milvus::GetPartitionStatisticsResponse>(
            pre, &MilvusConnection::GetPartitionStatistics, post);
}

Status
MilvusClientV2Impl::ListPartitions(const ListPartitionsRequest& request, ListPartitionsResponse& response) {
    auto pre = [&request](proto::milvus::ShowPartitionsRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_type(proto::milvus::ShowType::All);  // follow pymilvus behavior, always show all partitions
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::ShowPartitionsResponse& rpc_response) {
        std::vector<std::string> partition_names;
        std::vector<PartitionInfo> partition_infos;
        partition_names.reserve(rpc_response.partition_names_size());
        partition_infos.reserve(rpc_response.partition_names_size());
        for (int i = 0; i < rpc_response.partition_names_size(); ++i) {
            partition_names.push_back(rpc_response.partition_names(i));
            partition_infos.emplace_back(rpc_response.partition_names(i), rpc_response.partitionids(i),
                                         rpc_response.created_timestamps(i));
        }

        response.SetPartitionNames(std::move(partition_names));
        response.SetPartitionInfos(std::move(partition_infos));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::ShowPartitionsRequest, proto::milvus::ShowPartitionsResponse>(
        pre, &MilvusConnection::ShowPartitions, post);
}

Status
MilvusClientV2Impl::CreateAlias(const CreateAliasRequest& request) {
    auto pre = [&request](proto::milvus::CreateAliasRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_alias(request.Alias());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::CreateAliasRequest, proto::common::Status>(pre,
                                                                                        &MilvusConnection::CreateAlias);
}

Status
MilvusClientV2Impl::DropAlias(const DropAliasRequest& request) {
    auto pre = [&request](proto::milvus::DropAliasRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_alias(request.Alias());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::DropAliasRequest, proto::common::Status>(pre,
                                                                                      &MilvusConnection::DropAlias);
}

Status
MilvusClientV2Impl::AlterAlias(const AlterAliasRequest& request) {
    auto pre = [&request](proto::milvus::AlterAliasRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_alias(request.Alias());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::AlterAliasRequest, proto::common::Status>(pre,
                                                                                       &MilvusConnection::AlterAlias);
}

Status
MilvusClientV2Impl::DescribeAlias(const DescribeAliasRequest& request, DescribeAliasResponse& response) {
    auto pre = [&request](proto::milvus::DescribeAliasRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_alias(request.Alias());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::DescribeAliasResponse& rpc_response) {
        AliasDesc desc;
        desc.SetName(rpc_response.alias());
        desc.SetDatabaseName(rpc_response.db_name());
        desc.SetCollectionName(rpc_response.collection());

        response.SetDesc(std::move(desc));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::DescribeAliasRequest, proto::milvus::DescribeAliasResponse>(
        pre, &MilvusConnection::DescribeAlias, post);
}

Status
MilvusClientV2Impl::ListAliases(const ListAliasesRequest& request, ListAliasesResponse& response) {
    auto pre = [&request](proto::milvus::ListAliasesRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::ListAliasesResponse& rpc_response) {
        std::vector<std::string> aliases;
        aliases.reserve(rpc_response.aliases_size());
        for (auto i = 0; i < rpc_response.aliases_size(); i++) {
            aliases.push_back(rpc_response.aliases(i));
        }
        response.SetAliases(std::move(aliases));
        response.SetDatabaseName(rpc_response.db_name());
        response.SetCollectionName(rpc_response.collection_name());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::ListAliasesRequest, proto::milvus::ListAliasesResponse>(
        pre, &MilvusConnection::ListAliases, post);
}

Status
MilvusClientV2Impl::UseDatabase(const std::string& db_name) {
    cleanCollectionDescCache();
    return connection_.UseDatabase(db_name);
}

Status
MilvusClientV2Impl::CurrentUsedDatabase(std::string& db_name) {
    // the db name is returned from ConnectParam, the default db_name of ConnectParam
    // is an empty string which means the default database named "default".
    auto name = connection_.CurrentDbName("");
    db_name = name.empty() ? "default" : name;
    return Status::OK();
}

Status
MilvusClientV2Impl::CreateDatabase(const CreateDatabaseRequest& request) {
    auto pre = [&request](proto::milvus::CreateDatabaseRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());

        for (const auto& pair : request.Properties()) {
            auto kv_pair = rpc_request.add_properties();
            kv_pair->set_key(pair.first);
            kv_pair->set_value(pair.second);
        }
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::CreateDatabaseRequest, proto::common::Status>(
        pre, &MilvusConnection::CreateDatabase);
}

Status
MilvusClientV2Impl::DropDatabase(const DropDatabaseRequest& request) {
    auto pre = [&request](proto::milvus::DropDatabaseRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::DropDatabaseRequest, proto::common::Status>(
        pre, &MilvusConnection::DropDatabase);
}

Status
MilvusClientV2Impl::ListDatabases(const ListDatabasesRequest& request, ListDatabasesResponse& response) {
    auto post = [&response](const proto::milvus::ListDatabasesResponse& rpc_response) {
        std::vector<std::string> db_names;
        db_names.reserve(rpc_response.db_names_size());
        for (int i = 0; i < rpc_response.db_names_size(); i++) {
            db_names.push_back(rpc_response.db_names(i));
        }

        response.SetDatabaseNames(std::move(db_names));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::ListDatabasesRequest, proto::milvus::ListDatabasesResponse>(
        nullptr, &MilvusConnection::ListDatabases, post);
}

Status
MilvusClientV2Impl::AlterDatabaseProperties(const AlterDatabasePropertiesRequest& request) {
    auto pre = [&request](proto::milvus::AlterDatabaseRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());

        for (const auto& pair : request.Properties()) {
            auto kv_pair = rpc_request.add_properties();
            kv_pair->set_key(pair.first);
            kv_pair->set_value(pair.second);
        }
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::AlterDatabaseRequest, proto::common::Status>(
        pre, &MilvusConnection::AlterDatabase);
}

Status
MilvusClientV2Impl::DropDatabaseProperties(const DropDatabasePropertiesRequest& request) {
    auto pre = [&request](proto::milvus::AlterDatabaseRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());

        for (const auto& key : request.PropertyKeys()) {
            rpc_request.add_delete_keys(key);
        }
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::AlterDatabaseRequest, proto::common::Status>(
        pre, &MilvusConnection::AlterDatabase);
}

Status
MilvusClientV2Impl::DescribeDatabase(const DescribeDatabaseRequest& request, DescribeDatabaseResponse& response) {
    auto pre = [&request](proto::milvus::DescribeDatabaseRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::DescribeDatabaseResponse& rpc_response) {
        DatabaseDesc db_desc;
        db_desc.SetName(rpc_response.db_name());
        db_desc.SetID(rpc_response.dbid());
        db_desc.SetCreatedTime(rpc_response.created_timestamp());
        std::unordered_map<std::string, std::string> properties;
        for (int i = 0; i < rpc_response.properties_size(); i++) {
            const auto& prop = rpc_response.properties(i);
            properties[prop.key()] = prop.value();
        }
        db_desc.SetProperties(properties);

        response.SetDesc(std::move(db_desc));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::DescribeDatabaseRequest, proto::milvus::DescribeDatabaseResponse>(
        pre, &MilvusConnection::DescribeDatabase, post);
}

Status
MilvusClientV2Impl::CreateIndex(const CreateIndexRequest& request) {
    const auto& descs = request.Indexes();
    for (const auto& desc : descs) {
        auto status = createIndex(request.DatabaseName(), request.CollectionName(), desc, request.Sync());
        if (!status.IsOk()) {
            return status;
        }

        // TODO: check timeout value in sync mode
    }
    return Status::OK();
}

Status
MilvusClientV2Impl::DescribeIndex(const DescribeIndexRequest& request, DescribeIndexResponse& response) {
    auto pre = [&request](proto::milvus::DescribeIndexRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_field_name(request.FieldName());
        rpc_request.set_timestamp(request.Timestamp());
        return Status::OK();
    };

    auto post = [&request, &response](const proto::milvus::DescribeIndexResponse& rpc_response) {
        auto count = rpc_response.index_descriptions_size();
        if (!request.FieldName().empty() and count == 0) {
            return Status{StatusCode::SERVER_FAILED, "index not found:" + request.FieldName()};
        }

        std::vector<IndexDesc> descs;
        for (auto i = 0; i < count; i++) {
            auto rpc_desc = rpc_response.index_descriptions(0);
            IndexDesc index_desc;
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
            descs.emplace_back(std::move(index_desc));
        }

        response.SetDescs(std::move(descs));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::DescribeIndexRequest, proto::milvus::DescribeIndexResponse>(
        pre, &MilvusConnection::DescribeIndex, post);
}

Status
MilvusClientV2Impl::ListIndexes(const ListIndexesRequest& request, ListIndexesResponse& response) {
    DescribeIndexRequest d_request = DescribeIndexRequest()
                                         .WithDatabaseName(request.DatabaseName())
                                         .WithCollectionName(request.CollectionName())
                                         .WithFieldName("");
    DescribeIndexResponse d_response;
    auto status = DescribeIndex(d_request, d_response);
    if (!status.IsOk()) {
        return status;
    }

    std::vector<std::string> index_names;
    std::vector<IndexDesc> descs = d_response.Descs();
    for (const auto& desc : descs) {
        index_names.push_back(desc.IndexName());
    }
    response.SetDescs(std::move(descs));
    response.SetIndexNames(std::move(index_names));

    return Status::OK();
}

Status
MilvusClientV2Impl::DropIndex(const DropIndexRequest& request) {
    auto pre = [&request](proto::milvus::DropIndexRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_index_name(request.FieldName());
        return Status::OK();
    };
    return connection_.Invoke<proto::milvus::DropIndexRequest, proto::common::Status>(pre,
                                                                                      &MilvusConnection::DropIndex);
}

Status
MilvusClientV2Impl::AlterIndexProperties(const AlterIndexPropertiesRequest& request) {
    auto pre = [&request](proto::milvus::AlterIndexRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_index_name(request.FieldName());
        for (const auto& pair : request.Properties()) {
            auto kv_pair = rpc_request.add_extra_params();
            kv_pair->set_key(pair.first);
            kv_pair->set_value(pair.second);
        }

        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::AlterIndexRequest, proto::common::Status>(pre,
                                                                                       &MilvusConnection::AlterIndex);
}

Status
MilvusClientV2Impl::DropIndexProperties(const DropIndexPropertiesRequest& request) {
    auto pre = [&request](proto::milvus::AlterIndexRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_index_name(request.FieldName());
        for (const auto& name : request.PropertyKeys()) {
            rpc_request.add_delete_keys(name);
        }

        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::AlterIndexRequest, proto::common::Status>(pre,
                                                                                       &MilvusConnection::AlterIndex);
}

Status
MilvusClientV2Impl::Insert(const InsertRequest& request, InsertResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::Upsert(const UpsertRequest& request, UpsertResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::Delete(const DeleteRequest& request, DeleteResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::Search(const SearchRequest& request, SearchResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::SearchIterator(const SearchIteratorRequest& request, SearchIteratorPtr& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::HybridSearch(const HybridSearchRequest& request, HybridSearchResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::Query(const QueryRequest& request, QueryResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::QueryIterator(const QueryIteratorRequest& request, QueryIteratorPtr& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::RunAnalyzer(const RunAnalyzerRequest& request, RunAnalyzerResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::Flush(const FlushRequest& request) {
    return Status::OK();
}

Status
MilvusClientV2Impl::ListPersistentSegments(const ListPersistentSegmentsRequest& request,
                                           ListPersistentSegmentsResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::ListQuerySegments(const ListQuerySegmentsRequest& request, ListQuerySegmentsResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::Compact(const CompactRequest& request, CompactResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::GetCompactionState(const GetCompactionStateRequest& request, GetCompactionStateResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::GetCompactionPlans(const GetCompactionPlansRequest& request, GetCompactionPlansResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::CreateResourceGroup(const CreateResourceGroupRequest& request) {
    return Status::OK();
}

Status
MilvusClientV2Impl::DropResourceGroup(const DropResourceGroupRequest& request) {
    return Status::OK();
}

Status
MilvusClientV2Impl::UpdateResourceGroups(const UpdateResourceGroupsRequest& request) {
    return Status::OK();
}

Status
MilvusClientV2Impl::TransferNode(const TransferNodeRequest& request) {
    return Status::OK();
}

Status
MilvusClientV2Impl::TransferReplica(const TransferReplicaRequest& request) {
    return Status::OK();
}

Status
MilvusClientV2Impl::ListResourceGroups(const ListResourceGroupsRequest& request, ListResourceGroupsResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::DescribeResourceGroup(const DescribeResourceGroupRequest& request,
                                          DescribeResourceGroupResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::CreateUser(const CreateUserRequest& request) {
    return Status::OK();
}

Status
MilvusClientV2Impl::UpdatePassword(const UpdatePasswordRequest& request) {
    return Status::OK();
}

Status
MilvusClientV2Impl::DropUser(const DropUserRequest& request) {
    return Status::OK();
}

Status
MilvusClientV2Impl::DescribeUser(const DescribeUserRequest& request, DescribeUserResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::ListUsers(const ListUsersRequest& request, ListUsersResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::CreateRole(const CreateRoleRequest& request) {
    return Status::OK();
}

Status
MilvusClientV2Impl::DropRole(const DropRoleRequest& request) {
    return Status::OK();
}

Status
MilvusClientV2Impl::DescribeRole(const DescribeRoleRequest& request, DescribeRoleResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::ListRoles(const ListRolesRequest& request, ListRolesResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::GrantRole(const GrantRoleRequest& request) {
    return Status::OK();
}

Status
MilvusClientV2Impl::RevokeRole(const RevokeRoleRequest& request) {
    return Status::OK();
}

Status
MilvusClientV2Impl::GrantPrivilegeV2(const GrantPrivilegeV2Request& request) {
    return Status::OK();
}

Status
MilvusClientV2Impl::RevokePrivilegeV2(const RevokePrivilegeV2Request& request) {
    return Status::OK();
}

Status
MilvusClientV2Impl::CreatePrivilegeGroup(const CreatePrivilegeGroupRequest& request) {
    return Status::OK();
}

Status
MilvusClientV2Impl::DropPrivilegeGroup(const DropPrivilegeGroupRequest& request) {
    return Status::OK();
}

Status
MilvusClientV2Impl::ListPrivilegeGroups(const ListPrivilegeGroupsRequest& request,
                                        ListPrivilegeGroupsResponse& response) {
    return Status::OK();
}

Status
MilvusClientV2Impl::AddPrivilegesToGroup(const AddPrivilegesToGroupRequest& request) {
    return Status::OK();
}

Status
MilvusClientV2Impl::RemovePrivilegesFromGroup(const RemovePrivilegesFromGroupRequest& request) {
    return Status::OK();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// internal used methods
Status
MilvusClientV2Impl::createIndex(const std::string& db_name, const std::string& collection_name, const IndexDesc& desc,
                                bool sync) {
    auto pre = [&db_name, &collection_name, &desc](proto::milvus::CreateIndexRequest& rpc_request) {
        rpc_request.set_db_name(db_name);
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_field_name(desc.FieldName());
        rpc_request.set_index_name(desc.IndexName());

        auto kv_pair = rpc_request.add_extra_params();
        kv_pair->set_key(milvus::INDEX_TYPE);
        kv_pair->set_value(std::to_string(desc.IndexType()));

        // for scalar fields, no metric type
        if (desc.MetricType() != MetricType::DEFAULT) {
            kv_pair = rpc_request.add_extra_params();
            kv_pair->set_key(milvus::METRIC_TYPE);
            kv_pair->set_value(std::to_string(desc.MetricType()));
        }

        kv_pair = rpc_request.add_extra_params();
        kv_pair->set_key(milvus::PARAMS);
        ::nlohmann::json json_obj(desc.ExtraParams());
        kv_pair->set_value(json_obj.dump());

        return Status::OK();
    };

    // if not sync mode, directly return
    if (!sync) {
        return connection_.Invoke<proto::milvus::CreateIndexRequest, proto::common::Status>(
            pre, &MilvusConnection::CreateIndex);
    }

    ProgressMonitor progress_monitor = ProgressMonitor::Forever();
    auto wait_for_status = [&db_name, &collection_name, &desc, &progress_monitor, this](const proto::common::Status&) {
        return ConnectionHandler::WaitForStatus(
            [&db_name, &collection_name, &desc, this](Progress& progress) -> Status {
                DescribeIndexRequest request = DescribeIndexRequest()
                                                   .WithDatabaseName(db_name)
                                                   .WithCollectionName(collection_name)
                                                   .WithFieldName(desc.FieldName());
                DescribeIndexResponse response;
                auto status = DescribeIndex(request, response);
                if (!status.IsOk()) {
                    return status;
                }

                // each field only returns one index desc, but in future if we support multi-indexes in one filed,
                // describeIndex() might return multiple descs. now we only process the first desc.
                const auto& out_descs = response.Descs();
                if (out_descs.empty()) {
                    // server-side error, it should return one desc here
                    return Status{StatusCode::SERVER_FAILED, "Index is created by cannot be described"};
                }

                const auto& out_desc = out_descs.at(0);
                // if index finished, progress set to 100%
                // else if index failed, return error status
                // else if index is in progressing, continue to check
                if (out_desc.StateCode() == IndexStateCode::FINISHED || out_desc.StateCode() == IndexStateCode::NONE) {
                    progress.finished_ = 100;
                } else if (out_desc.StateCode() == IndexStateCode::FAILED) {
                    return Status{StatusCode::SERVER_FAILED, "index failed:" + out_desc.FailReason()};
                }

                return status;
            },
            progress_monitor);
    };
    return connection_.Invoke<proto::milvus::CreateIndexRequest, proto::common::Status>(
        nullptr, pre, &MilvusConnection::CreateIndex, wait_for_status, nullptr);
}

std::string
combineDbCollectionName(const std::string& db_name, const std::string& collection_name) {
    return std::string(db_name) + "|" + collection_name;
}

Status
MilvusClientV2Impl::getCollectionDesc(const std::string& db_name, const std::string& collection_name, bool force_update,
                                      CollectionDescPtr& desc_ptr) {
    // this lock locks the entire section, including the call of DescribeCollection()
    // the reason is: describeCollection() could be limited by server-side(DDL request throttling is enabled)
    // we don't intend to allow too many threads run into describeCollection() in this method
    std::lock_guard<std::mutex> lock(collection_desc_cache_mtx_);
    auto it = collection_desc_cache_.find(collection_name);
    if (it != collection_desc_cache_.end()) {
        if (it->second != nullptr && !force_update) {
            desc_ptr = it->second;
            return Status::OK();
        }
    }

    DescribeCollectionRequest rquest =
        DescribeCollectionRequest().WithDatabaseName(db_name).WithCollectionName(collection_name);
    DescribeCollectionResponse response;
    auto status = DescribeCollection(rquest, response);
    if (status.IsOk()) {
        desc_ptr = std::make_shared<CollectionDesc>(response.Desc());
        auto name = combineDbCollectionName(db_name, collection_name);
        collection_desc_cache_[name] = desc_ptr;
        return status;
    }
    return status;
}

void
MilvusClientV2Impl::cleanCollectionDescCache() {
    std::lock_guard<std::mutex> lock(collection_desc_cache_mtx_);
    collection_desc_cache_.clear();
}

void
MilvusClientV2Impl::removeCollectionDesc(const std::string& db_name, const std::string& collection_name) {
    auto name = combineDbCollectionName(db_name, collection_name);
    std::lock_guard<std::mutex> lock(collection_desc_cache_mtx_);
    collection_desc_cache_.erase(name);
}

}  // namespace milvus
