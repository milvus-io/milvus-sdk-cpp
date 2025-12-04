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

std::shared_ptr<MilvusClientV2>
MilvusClientV2::Create() {
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
    auto pre = [&request](proto::milvus::CheckHealthRequest& rpc_request) { return Status::OK(); };

    auto post = [&response](const proto::milvus::CheckHealthResponse& rpc_response) {
        response.SetIsHealthy(rpc_response.ishealthy());
        std::vector<std::string> reasons;
        reasons.reserve(rpc_response.reasons_size());
        for (auto i = 0; i < rpc_response.reasons_size(); i++) {
            reasons.push_back(rpc_response.reasons(i));
        }
        response.SetReasons(std::move(reasons));

        std::vector<std::string> quota_states;
        quota_states.reserve(rpc_response.quota_states_size());
        for (auto i = 0; i < rpc_response.quota_states_size(); i++) {
            quota_states.push_back(QuotaState_Name(rpc_response.quota_states(i)));
        }
        response.SetQuotaStates(std::move(quota_states));

        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::CheckHealthRequest, proto::milvus::CheckHealthResponse>(
        pre, &MilvusConnection::CheckHealth, post);
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
        rpc_request.set_replica_number(static_cast<int32_t>(request.ReplicaNum()));
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
                std::set<std::string> partition_names;
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
        rpc_request.set_replica_number(static_cast<int32_t>(request.ReplicaNum()));
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
        if (!request.FieldName().empty() && count == 0) {
            return Status{StatusCode::SERVER_FAILED, "Index not found:" + request.FieldName()};
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

    std::vector<IndexDesc> descs = d_response.Descs();
    std::vector<std::string> index_names;
    index_names.reserve(descs.size());
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
    CollectionDescPtr collection_desc;
    std::vector<proto::schema::FieldData> rpc_fields;
    auto validate = [this, &request, &collection_desc, &rpc_fields]() {
        auto status = getCollectionDesc(request.DatabaseName(), request.CollectionName(), false, collection_desc);
        if (!status.IsOk()) {
            return status;
        }

        const auto& fields = request.ColumnsData();
        const auto& rows = request.RowsData();
        if (!fields.empty() && !rows.empty()) {
            return Status{StatusCode::INVALID_AGUMENT, "Not allow to set ColumnsData and RowsData both"};
        }

        if (!rows.empty()) {
            // verify and convert row-based data to rpc fields
            status = CheckAndSetRowData(rows, collection_desc->Schema(), false, rpc_fields);
            if (!status.IsOk()) {
                return status;
            }
        } else if (!fields.empty()) {
            // verify column-based data
            // if the collection is already recreated, some schema might be changed, we need to update the
            // collectionDesc cache and call CheckInsertInput() again.
            status = CheckInsertInput(collection_desc, fields, false);
            if (status.Code() == milvus::StatusCode::DATA_UNMATCH_SCHEMA) {
                status = getCollectionDesc(request.DatabaseName(), request.CollectionName(), true, collection_desc);
                if (!status.IsOk()) {
                    return status;
                }

                status = CheckInsertInput(collection_desc, fields, false);
            }

            // convert column-based data to rpc fields
            status = CreateProtoFieldDatas(collection_desc->Schema(), fields, rpc_fields);
            if (!status.IsOk()) {
                return status;
            }
        }

        return Status::OK();
    };

    auto pre = [&request, &collection_desc, &rpc_fields](proto::milvus::InsertRequest& rpc_request) {
        const auto& fields = request.ColumnsData();
        const auto& rows = request.RowsData();
        auto row_count = rows.size();
        if (!fields.empty()) {
            row_count = (*fields.front()).Count();
        }

        auto* mutable_fields = rpc_request.mutable_fields_data();
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_partition_name(request.PartitionName());
        rpc_request.set_num_rows(static_cast<uint32_t>(row_count));
        rpc_request.set_schema_timestamp(collection_desc->UpdateTime());
        for (auto& field : rpc_fields) {
            mutable_fields->Add(std::move(field));
        }

        return Status::OK();
    };

    auto post = [this, &request, &response](const proto::milvus::MutationResult& rpc_response) {
        DmlResults results;
        auto id_array = CreateIDArray(rpc_response.ids());
        results.SetIdArray(std::move(id_array));
        results.SetTimestamp(rpc_response.timestamp());
        results.SetInsertCount(static_cast<uint64_t>(rpc_response.insert_cnt()));
        response.SetResults(std::move(results));

        // special for dml api: if the api failed, remove the schema cache of this collection
        if (IsRealFailure(rpc_response.status())) {
            removeCollectionDesc(request.DatabaseName(), request.CollectionName());
        } else {
            auto db_name = connection_.CurrentDbName(request.DatabaseName());
            GtsDict::GetInstance().UpdateCollectionTs(db_name, request.CollectionName(), rpc_response.timestamp());
        }

        return Status::OK();
    };

    auto status = connection_.Invoke<proto::milvus::InsertRequest, proto::milvus::MutationResult>(
        validate, pre, &MilvusConnection::Insert, post);
    // If there are multiple clients, the client_A repeatedly do insert, the client_B changes
    // the collection schema. The server might return a special error code "SchemaMismatch".
    // If the client_A gets this special error code, it needs to update the collectionDesc cache and
    // call Insert() again.
    if (status.LegacyServerCode() == static_cast<int32_t>(proto::common::ErrorCode::SchemaMismatch)) {
        removeCollectionDesc(request.DatabaseName(), request.CollectionName());
        return Insert(request, response);
    }
    return status;
}

Status
MilvusClientV2Impl::Upsert(const UpsertRequest& request, UpsertResponse& response) {
    std::vector<proto::schema::FieldData> rpc_fields;
    CollectionDescPtr collection_desc;
    auto validate = [this, &request, &collection_desc, &rpc_fields]() {
        auto status = getCollectionDesc(request.DatabaseName(), request.CollectionName(), false, collection_desc);
        if (!status.IsOk()) {
            return status;
        }

        const auto& fields = request.ColumnsData();
        const auto& rows = request.RowsData();
        if (!fields.empty() && !rows.empty()) {
            return Status{StatusCode::INVALID_AGUMENT, "Not allow to set ColumnsData and RowsData both"};
        }

        if (!rows.empty()) {
            // verify and convert row-based data to rpc fields
            status = CheckAndSetRowData(rows, collection_desc->Schema(), true, rpc_fields);
            if (!status.IsOk()) {
                return status;
            }
        } else if (!fields.empty()) {
            // verify column-based data
            // if the collection is already recreated, some schema might be changed, we need to update the
            // collectionDesc cache and call CheckInsertInput() again.
            status = CheckInsertInput(collection_desc, fields, true);
            if (status.Code() == milvus::StatusCode::DATA_UNMATCH_SCHEMA) {
                status = getCollectionDesc(request.DatabaseName(), request.CollectionName(), true, collection_desc);
                if (!status.IsOk()) {
                    return status;
                }

                status = CheckInsertInput(collection_desc, fields, true);
            }

            // convert column-based data to rpc fields
            status = CreateProtoFieldDatas(collection_desc->Schema(), fields, rpc_fields);
            if (!status.IsOk()) {
                return status;
            }
        }

        return Status::OK();
    };

    auto pre = [&request, &collection_desc, &rpc_fields](proto::milvus::UpsertRequest& rpc_request) {
        const auto& fields = request.ColumnsData();
        const auto& rows = request.RowsData();
        auto row_count = rows.size();
        if (!fields.empty()) {
            row_count = (*fields.front()).Count();
        }

        auto* mutable_fields = rpc_request.mutable_fields_data();
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_partition_name(request.PartitionName());
        rpc_request.set_num_rows(static_cast<uint32_t>(row_count));
        rpc_request.set_schema_timestamp(collection_desc->UpdateTime());
        for (auto& field : rpc_fields) {
            mutable_fields->Add(std::move(field));
        }

        return Status::OK();
    };

    auto post = [this, &request, &response](const proto::milvus::MutationResult& rpc_response) {
        DmlResults results;
        auto id_array = CreateIDArray(rpc_response.ids());
        results.SetIdArray(std::move(id_array));
        results.SetTimestamp(rpc_response.timestamp());
        results.SetUpsertCount(static_cast<uint64_t>(rpc_response.upsert_cnt()));
        response.SetResults(std::move(results));

        // special for dml api: if the api failed, remove the schema cache of this collection
        if (IsRealFailure(rpc_response.status())) {
            removeCollectionDesc(request.DatabaseName(), request.CollectionName());
        } else {
            auto db_name = connection_.CurrentDbName(request.DatabaseName());
            GtsDict::GetInstance().UpdateCollectionTs(db_name, request.CollectionName(), rpc_response.timestamp());
        }
        return Status::OK();
    };

    auto status = connection_.Invoke<proto::milvus::UpsertRequest, proto::milvus::MutationResult>(
        validate, pre, &MilvusConnection::Upsert, post);
    // If there are multiple clients, the client_A repeatedly do insert, the client_B changes
    // the collection schema. The server might return a special error code "SchemaMismatch".
    // If the client_A gets this special error code, it needs to update the collectionDesc cache and
    // call Upsert() again.
    if (status.LegacyServerCode() == static_cast<int32_t>(proto::common::ErrorCode::SchemaMismatch)) {
        removeCollectionDesc(request.DatabaseName(), request.CollectionName());
        return Upsert(request, response);
    }
    return status;
}

Status
MilvusClientV2Impl::Delete(const DeleteRequest& request, DeleteResponse& response) {
    auto pre = [this, &request](proto::milvus::DeleteRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_partition_name(request.PartitionName());

        if (request.Filter().empty() && request.IDs().GetRowCount() == 0) {
            return Status{StatusCode::INVALID_AGUMENT,
                          "Deletion condition must be specified, by primary keys or by filter expression"};
        }

        if (!request.Filter().empty() && request.IDs().GetRowCount() != 0) {
            return Status{StatusCode::INVALID_AGUMENT,
                          "Ambiguous filter parameter, only one deletion condition can be specified"};
        }

        if (!request.Filter().empty()) {
            // delete by filter expression
            rpc_request.set_expr(request.Filter());
            auto rpc_templates = rpc_request.mutable_expr_template_values();
            const auto& templates = request.FilterTemplates();
            auto status = ConvertFilterTemplates(templates, rpc_templates);
            if (!status.IsOk()) {
                return status;
            }
        } else if (request.IDs().GetRowCount() != 0) {
            // delete by ids, we need the collection schema to get primary key name
            CollectionDescPtr collection_desc;
            auto status = getCollectionDesc(request.DatabaseName(), request.CollectionName(), false, collection_desc);
            if (!status.IsOk()) {
                return status;
            }

            // use filter template to pass the id array
            auto pk = collection_desc->Schema().PrimaryFieldName();
            rpc_request.set_expr(pk + " in {ids}");
            std::unordered_map<std::string, nlohmann::json> templates;
            if (request.IDs().IsIntegerID()) {
                templates.insert(std::make_pair("ids", request.IDs().IntIDArray()));
            } else {
                templates.insert(std::make_pair("ids", request.IDs().StrIDArray()));
            }

            auto rpc_templates = rpc_request.mutable_expr_template_values();
            status = ConvertFilterTemplates(templates, rpc_templates);
            if (!status.IsOk()) {
                return status;
            }
        }

        return Status::OK();
    };

    auto post = [this, &request, &response](const proto::milvus::MutationResult& rpc_response) {
        DmlResults results;
        auto id_array = CreateIDArray(rpc_response.ids());
        results.SetIdArray(std::move(id_array));
        results.SetTimestamp(rpc_response.timestamp());
        results.SetDeleteCount(static_cast<uint64_t>(rpc_response.delete_cnt()));
        response.SetResults(std::move(results));

        if (!IsRealFailure(rpc_response.status())) {
            auto db_name = connection_.CurrentDbName(request.DatabaseName());
            GtsDict::GetInstance().UpdateCollectionTs(db_name, request.CollectionName(), rpc_response.timestamp());
        }
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::DeleteRequest, proto::milvus::MutationResult>(
        pre, &MilvusConnection::Delete, post);
}

Status
MilvusClientV2Impl::Search(const SearchRequest& request, SearchResponse& response) {
    auto validate = [&request]() {
        if (request.TargetVectors() == nullptr || request.TargetVectors()->Count() == 0) {
            return Status{StatusCode::INVALID_AGUMENT, "No target vector is assigned"};
        }

        return Status::OK();
    };

    auto pre = [this, &request](proto::milvus::SearchRequest& rpc_request) {
        auto current_name = connection_.CurrentDbName(request.DatabaseName());
        ConvertSearchRequest<SearchRequest>(request, current_name, rpc_request);
        return Status::OK();
    };

    auto post = [this, &request, &response](const proto::milvus::SearchResults& rpc_response) {
        // in milvus version older than v2.4.20, the primary_field_name() is empty, we need to
        // get the primary key field name from collection schema
        SearchResults results;
        const auto& result_data = rpc_response.results();
        auto pk_name = result_data.primary_field_name();
        if (result_data.primary_field_name().empty()) {
            CollectionDescPtr collection_desc;
            getCollectionDesc(request.DatabaseName(), request.CollectionName(), false, collection_desc);
            if (collection_desc != nullptr) {
                pk_name = collection_desc->Schema().Name();
            }
        }
        auto status = ConvertSearchResults(rpc_response, pk_name, results);
        response.SetResults(std::move(results));
        return status;
    };

    return connection_.Invoke<proto::milvus::SearchRequest, proto::milvus::SearchResults>(
        validate, pre, &MilvusConnection::Search, nullptr, post);
}

Status
MilvusClientV2Impl::SearchIterator(SearchIteratorRequest& request, SearchIteratorPtr& iterator) {
    auto status = iteratorPrepare(request);
    if (!status.IsOk()) {
        return status;
    }

    // special process for search iterator
    // iterator needs vector field's metric type to determine the search range,
    // if user didn't offer the metric type, we need to describe the vector's index
    // to get the metric type.
    if (request.MetricType() == MetricType::DEFAULT) {
        std::string anns_field = request.AnnsField();
        if (anns_field.empty()) {
            CollectionDescPtr collection_desc;
            auto status = getCollectionDesc(request.DatabaseName(), request.CollectionName(), false, collection_desc);
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
                return {StatusCode::UNKNOWN_ERROR, "There should be at least one vector field in milvus collection"};
            }
            if (vector_field_names.size() > 1) {
                return {StatusCode::UNKNOWN_ERROR, "Must specify anns_field when there are more than one vector field"};
            }
            anns_field = *(vector_field_names.begin());
        }

        DescribeIndexRequest d_request = DescribeIndexRequest()
                                             .WithDatabaseName(request.DatabaseName())
                                             .WithCollectionName(request.CollectionName())
                                             .WithFieldName(anns_field);
        DescribeIndexResponse d_response;
        auto status = DescribeIndex(d_request, d_response);
        if (!status.IsOk()) {
            return status;
        }

        if (d_response.Descs().empty()) {
            return {StatusCode::UNKNOWN_ERROR, "Index not found: " + anns_field};
        }

        IndexDesc desc = d_response.Descs().at(0);
        request.SetMetricType(desc.MetricType());
    }

    // From SDK v2.5.6, milvus provide a new search iterator implementation in server-side.
    // SearchIteratorV2 is faster than V1 by 20~30 percent, and the recall is a little better than V1.
    // sdk attempts to use SearchIteratorV2 if supported by the server, otherwise falls back to V1.
    auto ptrV2 = std::make_shared<SearchIteratorV2Impl<SearchIteratorRequest>>(connection_.GetConnection(), request,
                                                                               connection_.GetRetryParam());
    status = ptrV2->Init();
    iterator = ptrV2;
    if (!status.IsOk() && status.Code() == StatusCode::NOT_SUPPORTED) {
        auto ptrV1 = std::make_shared<SearchIteratorImpl<SearchIteratorRequest>>(connection_.GetConnection(), request,
                                                                                 connection_.GetRetryParam());
        status = ptrV1->Init();
        if (!status.IsOk()) {
            return {status.Code(), "Unable to create search iterator, error: " + status.Message()};
        }
        iterator = ptrV1;
    }
    return status;
}

Status
MilvusClientV2Impl::HybridSearch(const HybridSearchRequest& request, HybridSearchResponse& response) {
    auto pre = [this, &request](proto::milvus::HybridSearchRequest& rpc_request) {
        auto current_name = connection_.CurrentDbName(request.DatabaseName());
        ConvertHybridSearchRequest<HybridSearchRequest>(request, current_name, rpc_request);
        return Status::OK();
    };

    auto post = [this, &request, &response](const proto::milvus::SearchResults& rpc_response) {
        // in milvus version older than v2.4.20, the primary_field_name() is empty, we need to
        // get the primary key field name from collection schema
        SearchResults results;
        const auto& result_data = rpc_response.results();
        auto pk_name = result_data.primary_field_name();
        if (result_data.primary_field_name().empty()) {
            CollectionDescPtr collection_desc;
            getCollectionDesc(request.DatabaseName(), request.CollectionName(), false, collection_desc);
            if (collection_desc != nullptr) {
                pk_name = collection_desc->Schema().Name();
            }
        }
        auto status = ConvertSearchResults(rpc_response, pk_name, results);
        response.SetResults(std::move(results));
        return status;
    };

    return connection_.Invoke<proto::milvus::HybridSearchRequest, proto::milvus::SearchResults>(
        pre, &MilvusConnection::HybridSearch, post);
}

Status
MilvusClientV2Impl::Query(const QueryRequest& request, QueryResponse& response) {
    auto pre = [this, &request](proto::milvus::QueryRequest& rpc_request) {
        auto current_name = connection_.CurrentDbName(request.DatabaseName());
        ConvertQueryRequest<QueryRequest>(request, current_name, rpc_request);
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::QueryResults& rpc_response) {
        QueryResults results;
        auto status = ConvertQueryResults(rpc_response, results);
        response.SetResults(std::move(results));
        return status;
    };

    return connection_.Invoke<proto::milvus::QueryRequest, proto::milvus::QueryResults>(pre, &MilvusConnection::Query,
                                                                                        post);
}

Status
MilvusClientV2Impl::QueryIterator(QueryIteratorRequest& request, QueryIteratorPtr& iterator) {
    auto status = iteratorPrepare(request);
    if (!status.IsOk()) {
        return status;
    }

    // iterator constructor might return error when it fails to initialize
    auto ptr = std::make_shared<QueryIteratorImpl<QueryIteratorRequest>>(connection_.GetConnection(), request,
                                                                         connection_.GetRetryParam());
    status = ptr->Init();
    if (!status.IsOk()) {
        return {status.Code(), "Unable to create query iterator, error: " + status.Message()};
    }
    iterator = ptr;
    return Status::OK();
}

Status
MilvusClientV2Impl::RunAnalyzer(const RunAnalyzerRequest& request, RunAnalyzerResponse& response) {
    auto pre = [&request](proto::milvus::RunAnalyzerRequest& rpc_request) {
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_field_name(request.FieldName());
        rpc_request.set_analyzer_params(request.AnalyzerParams().dump());
        auto placeholder = rpc_request.mutable_placeholder();
        for (std::string text : request.Texts()) {
            placeholder->Add(std::move(text));
        }
        for (const auto& name : request.AnalyzerNames()) {
            rpc_request.add_analyzer_names(name);
        }
        rpc_request.set_with_detail(request.IsWithDetail());
        rpc_request.set_with_hash(request.IsWithHash());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::RunAnalyzerResponse& rpc_response) {
        AnalyzerResults results;
        const auto& rpc_results = rpc_response.results();
        for (auto i = 0; i < rpc_response.results_size(); i++) {
            std::vector<AnalyzerToken> tokens;
            const auto& rpc_tokens = rpc_results[i].tokens();
            for (auto k = 0; k < rpc_results[i].tokens_size(); k++) {
                const auto& rpc_token = rpc_tokens[k];
                AnalyzerToken token;
                token.token_ = rpc_token.token();
                token.start_offset_ = rpc_token.start_offset();
                token.end_offset_ = rpc_token.end_offset();
                token.position_ = rpc_token.position();
                token.position_length_ = rpc_token.position_length();
                token.hash_ = rpc_token.hash();
                tokens.emplace_back(std::move(token));
            }

            AnalyzerResult result{std::move(tokens)};
            results.emplace_back(std::move(result));
        }
        response.SetResults(std::move(results));

        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::RunAnalyzerRequest, proto::milvus::RunAnalyzerResponse>(
        pre, &MilvusConnection::RunAnalyzer, post);
}

Status
MilvusClientV2Impl::Flush(const FlushRequest& request) {
    auto pre = [&request](proto::milvus::FlushRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        for (const auto& collection_name : request.CollectionNames()) {
            rpc_request.add_collection_names(collection_name);
        }
        return Status::OK();
    };

    // TODO: check timeout value in sync mode
    ProgressMonitor progress_monitor = ProgressMonitor::Forever();
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
            segment_count += static_cast<uint32_t>(pair.second.size());
        }
        if (segment_count == 0) {
            return Status::OK();
        }

        return ConnectionHandler::WaitForStatus(
            [&segment_count, &flush_segments, &finished_count, this](Progress& p) -> Status {
                p.total_ = segment_count;

                // call GetFlushState() to check segment state
                for (auto iter = flush_segments.begin(); iter != flush_segments.end();) {
                    bool flushed = false;
                    Status status = getFlushState(iter->second, flushed);
                    if (!status.IsOk()) {
                        return status;
                    }

                    if (flushed) {
                        finished_count += static_cast<uint32_t>(iter->second.size());
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

    return connection_.Invoke<proto::milvus::FlushRequest, proto::milvus::FlushResponse>(
        nullptr, pre, &MilvusConnection::Flush, wait_for_status, nullptr);
}

Status
MilvusClientV2Impl::ListPersistentSegments(const ListPersistentSegmentsRequest& request,
                                           ListPersistentSegmentsResponse& response) {
    auto pre = [&request](proto::milvus::GetPersistentSegmentInfoRequest& rpc_request) {
        rpc_request.set_dbname(request.DatabaseName());
        rpc_request.set_collectionname(request.CollectionName());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::GetPersistentSegmentInfoResponse& rpc_response) {
        SegmentsInfo segments_info;
        segments_info.reserve(rpc_response.infos_size());
        for (const auto& info : rpc_response.infos()) {
            segments_info.emplace_back(info.collectionid(), info.partitionid(), info.segmentid(), info.num_rows(),
                                       SegmentStateCast(info.state()));
        }
        response.SetResult(std::move(segments_info));
        return Status::OK();
    };

    return connection_
        .Invoke<proto::milvus::GetPersistentSegmentInfoRequest, proto::milvus::GetPersistentSegmentInfoResponse>(
            pre, &MilvusConnection::GetPersistentSegmentInfo, post);
}

Status
MilvusClientV2Impl::ListQuerySegments(const ListQuerySegmentsRequest& request, ListQuerySegmentsResponse& response) {
    auto pre = [&request](proto::milvus::GetQuerySegmentInfoRequest& rpc_request) {
        rpc_request.set_dbname(request.DatabaseName());
        rpc_request.set_collectionname(request.CollectionName());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::GetQuerySegmentInfoResponse& rpc_response) {
        QuerySegmentsInfo segments_info;
        segments_info.reserve(rpc_response.infos_size());
        for (const auto& info : rpc_response.infos()) {
            std::vector<int64_t> ids;
            for (auto id : info.nodeids()) {
                ids.push_back(id);
            }
            segments_info.emplace_back(info.collectionid(), info.partitionid(), info.segmentid(), info.num_rows(),
                                       milvus::SegmentStateCast(info.state()), info.index_name(), info.indexid(), ids);
        }
        response.SetResult(std::move(segments_info));
        return Status::OK();
    };
    return connection_.Invoke<proto::milvus::GetQuerySegmentInfoRequest, proto::milvus::GetQuerySegmentInfoResponse>(
        pre, &MilvusConnection::GetQuerySegmentInfo, post);
}

Status
MilvusClientV2Impl::Compact(const CompactRequest& request, CompactResponse& response) {
    CollectionDescPtr collection_desc;
    auto status = getCollectionDesc(request.DatabaseName(), request.CollectionName(), false, collection_desc);
    if (!status.IsOk()) {
        return status;
    }

    auto pre = [&collection_desc](proto::milvus::ManualCompactionRequest& rpc_request) {
        rpc_request.set_collectionid(collection_desc->ID());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::ManualCompactionResponse& rpc_response) {
        response.SetCompactionID(rpc_response.compactionid());
        response.SetCompactionPlanCount(rpc_response.compactionplancount());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::ManualCompactionRequest, proto::milvus::ManualCompactionResponse>(
        pre, &MilvusConnection::ManualCompaction, post);
}

Status
MilvusClientV2Impl::GetCompactionState(const GetCompactionStateRequest& request, GetCompactionStateResponse& response) {
    auto pre = [&request](proto::milvus::GetCompactionStateRequest& rpc_request) {
        rpc_request.set_compactionid(request.CompactionID());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::GetCompactionStateResponse& rpc_response) {
        CompactionState compaction_state;
        compaction_state.SetExecutingPlan(rpc_response.executingplanno());
        compaction_state.SetTimeoutPlan(rpc_response.timeoutplanno());
        compaction_state.SetCompletedPlan(rpc_response.completedplanno());
        compaction_state.SetFailedPlan(rpc_response.failedplanno());
        switch (rpc_response.state()) {
            case proto::common::CompactionState::Completed:
                compaction_state.SetState(CompactionStateCode::COMPLETED);
                break;
            case proto::common::CompactionState::Executing:
                compaction_state.SetState(CompactionStateCode::EXECUTING);
                break;
            default:
                break;
        }
        response.SetState(compaction_state);
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::GetCompactionStateRequest, proto::milvus::GetCompactionStateResponse>(
        pre, &MilvusConnection::GetCompactionState, post);
}

Status
MilvusClientV2Impl::GetCompactionPlans(const GetCompactionPlansRequest& request, GetCompactionPlansResponse& response) {
    auto pre = [&request](proto::milvus::GetCompactionPlansRequest& rpc_request) {
        rpc_request.set_compactionid(request.CompactionID());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::GetCompactionPlansResponse& rpc_response) {
        CompactionPlans plans;
        plans.reserve(rpc_response.mergeinfos_size());
        for (int i = 0; i < rpc_response.mergeinfos_size(); ++i) {
            auto& info = rpc_response.mergeinfos(i);
            std::vector<int64_t> source_ids;
            source_ids.reserve(info.sources_size());
            source_ids.insert(source_ids.end(), info.sources().begin(), info.sources().end());
            plans.emplace_back(source_ids, info.target());
        }
        response.SetPlans(std::move(plans));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::GetCompactionPlansRequest, proto::milvus::GetCompactionPlansResponse>(
        pre, &MilvusConnection::GetCompactionPlans, post);
}

Status
MilvusClientV2Impl::CreateResourceGroup(const CreateResourceGroupRequest& request) {
    auto pre = [&request](proto::milvus::CreateResourceGroupRequest& rpc_request) {
        rpc_request.set_resource_group(request.Name());

        auto rpc_config = new proto::rg::ResourceGroupConfig{};
        ConvertResourceGroupConfig(request.Config(), rpc_config);
        rpc_request.set_allocated_config(rpc_config);
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::CreateResourceGroupRequest, proto::common::Status>(
        pre, &MilvusConnection::CreateResourceGroup, nullptr);
}

Status
MilvusClientV2Impl::DropResourceGroup(const DropResourceGroupRequest& request) {
    auto pre = [&request](proto::milvus::DropResourceGroupRequest& rpc_request) {
        rpc_request.set_resource_group(request.GroupName());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::DropResourceGroupRequest, proto::common::Status>(
        pre, &MilvusConnection::DropResourceGroup, nullptr);
}

Status
MilvusClientV2Impl::UpdateResourceGroups(const UpdateResourceGroupsRequest& request) {
    auto pre = [&request](proto::milvus::UpdateResourceGroupsRequest& rpc_request) {
        for (const auto& pair : request.Groups()) {
            proto::rg::ResourceGroupConfig rpc_config;
            ConvertResourceGroupConfig(pair.second, &rpc_config);
            rpc_request.mutable_resource_groups()->insert(std::make_pair(pair.first, rpc_config));
        }
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::UpdateResourceGroupsRequest, proto::common::Status>(
        pre, &MilvusConnection::UpdateResourceGroups, nullptr);
}

Status
MilvusClientV2Impl::TransferNode(const TransferNodeRequest& request) {
    auto pre = [&request](proto::milvus::TransferNodeRequest& rpc_request) {
        rpc_request.set_source_resource_group(request.SourceGroup());
        rpc_request.set_target_resource_group(request.TargetGroup());
        rpc_request.set_num_node(static_cast<int32_t>(request.NumNodes()));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::TransferNodeRequest, proto::common::Status>(
        pre, &MilvusConnection::TransferNode, nullptr);
}

Status
MilvusClientV2Impl::TransferReplica(const TransferReplicaRequest& request) {
    auto pre = [&request](proto::milvus::TransferReplicaRequest& rpc_request) {
        rpc_request.set_source_resource_group(request.SourceGroup());
        rpc_request.set_target_resource_group(request.TargetGroup());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_num_replica(request.NumReplicas());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::TransferReplicaRequest, proto::common::Status>(
        pre, &MilvusConnection::TransferReplica, nullptr);
}

Status
MilvusClientV2Impl::ListResourceGroups(const ListResourceGroupsRequest& request, ListResourceGroupsResponse& response) {
    auto post = [&response](const proto::milvus::ListResourceGroupsResponse& rpc_response) {
        std::vector<std::string> group_names;
        group_names.reserve(rpc_response.resource_groups_size());
        for (const auto& group : rpc_response.resource_groups()) {
            group_names.push_back(group);
        }
        response.SetGroupNames(std::move(group_names));
        return Status::OK();
    };
    return connection_.Invoke<proto::milvus::ListResourceGroupsRequest, proto::milvus::ListResourceGroupsResponse>(
        nullptr, &MilvusConnection::ListResourceGroups, post);
}

Status
MilvusClientV2Impl::DescribeResourceGroup(const DescribeResourceGroupRequest& request,
                                          DescribeResourceGroupResponse& response) {
    auto pre = [&request](proto::milvus::DescribeResourceGroupRequest& rpc_request) {
        rpc_request.set_resource_group(request.GroupName());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::DescribeResourceGroupResponse& rpc_response) {
        ResourceGroupDesc desc;
        const auto& group = rpc_response.resource_group();
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
        response.SetDesc(std::move(desc));

        return Status::OK();
    };
    return connection_
        .Invoke<proto::milvus::DescribeResourceGroupRequest, proto::milvus::DescribeResourceGroupResponse>(
            pre, &MilvusConnection::DescribeResourceGroup, post);
}

Status
MilvusClientV2Impl::CreateUser(const CreateUserRequest& request) {
    auto pre = [&request](proto::milvus::CreateCredentialRequest& rpc_request) {
        rpc_request.set_username(request.UserName());
        rpc_request.set_password(milvus::Base64Encode(request.Password()));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::CreateCredentialRequest, proto::common::Status>(
        pre, &MilvusConnection::CreateCredential, nullptr);
}

Status
MilvusClientV2Impl::UpdatePassword(const UpdatePasswordRequest& request) {
    auto pre = [&request](proto::milvus::UpdateCredentialRequest& rpc_request) {
        rpc_request.set_username(request.UserName());
        rpc_request.set_oldpassword(milvus::Base64Encode(request.OldPassword()));
        rpc_request.set_newpassword(milvus::Base64Encode(request.NewPassword()));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::UpdateCredentialRequest, proto::common::Status>(
        pre, &MilvusConnection::UpdateCredential, nullptr);
}

Status
MilvusClientV2Impl::DropUser(const DropUserRequest& request) {
    auto pre = [&request](proto::milvus::DeleteCredentialRequest& rpc_request) {
        rpc_request.set_username(request.UserName());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::DeleteCredentialRequest, proto::common::Status>(
        pre, &MilvusConnection::DeleteCredential, nullptr);
}

Status
MilvusClientV2Impl::DescribeUser(const DescribeUserRequest& request, DescribeUserResponse& response) {
    auto pre = [&request](proto::milvus::SelectUserRequest& rpc_request) {
        rpc_request.mutable_user()->set_name(request.UserName());
        rpc_request.set_include_role_info(true);
        return Status::OK();
    };

    auto post = [&request, &response](const proto::milvus::SelectUserResponse& rpc_response) {
        UserDesc desc;
        desc.SetName(request.UserName());
        if (rpc_response.results().size() > 0) {
            auto result = rpc_response.results().at(0);
            for (const auto& role : result.roles()) {
                desc.AddRole(role.name());
            }
        }
        response.SetDesc(std::move(desc));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::SelectUserRequest, proto::milvus::SelectUserResponse>(
        pre, &MilvusConnection::SelectUser, post);
}

Status
MilvusClientV2Impl::ListUsers(const ListUsersRequest& request, ListUsersResponse& response) {
    auto post = [&response](const proto::milvus::ListCredUsersResponse& rpc_response) {
        std::vector<std::string> names;
        names.reserve(rpc_response.usernames_size());
        for (const auto& user : rpc_response.usernames()) {
            names.emplace_back(user);
        }
        response.SetUserNames(std::move(names));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::ListCredUsersRequest, proto::milvus::ListCredUsersResponse>(
        nullptr, &MilvusConnection::ListCredUsers, post);
}

Status
MilvusClientV2Impl::CreateRole(const CreateRoleRequest& request) {
    auto pre = [&request](proto::milvus::CreateRoleRequest& rpc_request) {
        rpc_request.mutable_entity()->set_name(request.RoleName());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::CreateRoleRequest, proto::common::Status>(
        pre, &MilvusConnection::CreateRole, nullptr);
}

Status
MilvusClientV2Impl::DropRole(const DropRoleRequest& request) {
    auto pre = [&request](proto::milvus::DropRoleRequest& rpc_request) {
        rpc_request.set_role_name(request.RoleName());
        rpc_request.set_force_drop(request.ForceDrop());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::DropRoleRequest, proto::common::Status>(pre, &MilvusConnection::DropRole,
                                                                                     nullptr);
}

Status
MilvusClientV2Impl::DescribeRole(const DescribeRoleRequest& request, DescribeRoleResponse& response) {
    auto pre = [&request](proto::milvus::SelectGrantRequest& rpc_request) {
        auto entity = rpc_request.mutable_entity();
        entity->mutable_role()->set_name(request.RoleName());
        entity->set_db_name(request.DatabaseName());
        return Status::OK();
    };

    auto post = [&request, &response](const proto::milvus::SelectGrantResponse& rpc_response) {
        RoleDesc desc;
        desc.SetName(request.RoleName());
        for (const auto& entity : rpc_response.entities()) {
            desc.AddGrantItem({entity.object().name(), entity.object_name(), entity.db_name(), entity.role().name(),
                               entity.grantor().user().name(), entity.grantor().privilege().name()});
        }
        response.SetDesc(std::move(desc));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::SelectGrantRequest, proto::milvus::SelectGrantResponse>(
        pre, &MilvusConnection::SelectGrant, post);
}

Status
MilvusClientV2Impl::ListRoles(const ListRolesRequest& request, ListRolesResponse& response) {
    auto pre = [](proto::milvus::SelectRoleRequest& rpc_request) {
        rpc_request.set_include_user_info(false);
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::SelectRoleResponse& rpc_response) {
        std::vector<std::string> names;
        names.reserve(rpc_response.results_size());
        for (const auto& result : rpc_response.results()) {
            names.push_back(result.role().name());
        }
        response.SetRoleNames(std::move(names));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::SelectRoleRequest, proto::milvus::SelectRoleResponse>(
        pre, &MilvusConnection::SelectRole, post);
}

Status
MilvusClientV2Impl::GrantRole(const GrantRoleRequest& request) {
    auto pre = [&request](proto::milvus::OperateUserRoleRequest& rpc_request) {
        rpc_request.set_username(request.UserName());
        rpc_request.set_role_name(request.RoleName());
        rpc_request.set_type(proto::milvus::OperateUserRoleType::AddUserToRole);
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::OperateUserRoleRequest, proto::common::Status>(
        pre, &MilvusConnection::OperateUserRole, nullptr);
}

Status
MilvusClientV2Impl::RevokeRole(const RevokeRoleRequest& request) {
    auto pre = [&request](proto::milvus::OperateUserRoleRequest& rpc_request) {
        rpc_request.set_username(request.UserName());
        rpc_request.set_role_name(request.RoleName());
        rpc_request.set_type(proto::milvus::OperateUserRoleType::RemoveUserFromRole);
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::OperateUserRoleRequest, proto::common::Status>(
        pre, &MilvusConnection::OperateUserRole, nullptr);
}

Status
MilvusClientV2Impl::GrantPrivilegeV2(const GrantPrivilegeV2Request& request) {
    auto pre = [&request](proto::milvus::OperatePrivilegeV2Request& rpc_request) {
        rpc_request.mutable_role()->set_name(request.RoleName());
        rpc_request.mutable_grantor()->mutable_privilege()->set_name(request.Privilege());
        rpc_request.set_type(proto::milvus::OperatePrivilegeType::Grant);
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_db_name(request.DatabaseName());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::OperatePrivilegeV2Request, proto::common::Status>(
        pre, &MilvusConnection::OperatePrivilegeV2, nullptr);
}

Status
MilvusClientV2Impl::RevokePrivilegeV2(const RevokePrivilegeV2Request& request) {
    auto pre = [&request](proto::milvus::OperatePrivilegeV2Request& rpc_request) {
        rpc_request.mutable_role()->set_name(request.RoleName());
        rpc_request.mutable_grantor()->mutable_privilege()->set_name(request.Privilege());
        rpc_request.set_type(proto::milvus::OperatePrivilegeType::Revoke);
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_db_name(request.DatabaseName());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::OperatePrivilegeV2Request, proto::common::Status>(
        pre, &MilvusConnection::OperatePrivilegeV2, nullptr);
}

Status
MilvusClientV2Impl::CreatePrivilegeGroup(const CreatePrivilegeGroupRequest& request) {
    auto pre = [&request](proto::milvus::CreatePrivilegeGroupRequest& rpc_request) {
        rpc_request.set_group_name(request.GroupName());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::CreatePrivilegeGroupRequest, proto::common::Status>(
        pre, &MilvusConnection::CreatePrivilegeGroup, nullptr);
}

Status
MilvusClientV2Impl::DropPrivilegeGroup(const DropPrivilegeGroupRequest& request) {
    auto pre = [&request](proto::milvus::DropPrivilegeGroupRequest& rpc_request) {
        rpc_request.set_group_name(request.GroupName());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::DropPrivilegeGroupRequest, proto::common::Status>(
        pre, &MilvusConnection::DropPrivilegeGroup, nullptr);
}

Status
MilvusClientV2Impl::ListPrivilegeGroups(const ListPrivilegeGroupsRequest& request,
                                        ListPrivilegeGroupsResponse& response) {
    auto post = [&response](const proto::milvus::ListPrivilegeGroupsResponse& rpc_response) {
        PrivilegeGroupInfos groups;
        groups.reserve(rpc_response.privilege_groups_size());
        for (const auto& result : rpc_response.privilege_groups()) {
            std::vector<std::string> privileges;
            for (const auto& rpc_privilege : result.privileges()) {
                privileges.push_back(rpc_privilege.name());
            }
            groups.emplace_back(result.group_name(), std::move(privileges));
        }
        response.SetGroups(std::move(groups));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::ListPrivilegeGroupsRequest, proto::milvus::ListPrivilegeGroupsResponse>(
        nullptr, &MilvusConnection::ListPrivilegeGroups, post);
}

Status
MilvusClientV2Impl::AddPrivilegesToGroup(const AddPrivilegesToGroupRequest& request) {
    auto pre = [&request](proto::milvus::OperatePrivilegeGroupRequest& rpc_request) {
        rpc_request.set_group_name(request.GroupName());
        for (const auto& privilege : request.Privileges()) {
            auto rpc_privilege = rpc_request.mutable_privileges()->Add();
            rpc_privilege->set_name(privilege);
        }
        rpc_request.set_type(proto::milvus::OperatePrivilegeGroupType::AddPrivilegesToGroup);
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::OperatePrivilegeGroupRequest, proto::common::Status>(
        pre, &MilvusConnection::OperatePrivilegeGroup, nullptr);
}

Status
MilvusClientV2Impl::RemovePrivilegesFromGroup(const RemovePrivilegesFromGroupRequest& request) {
    auto pre = [&request](proto::milvus::OperatePrivilegeGroupRequest& rpc_request) {
        rpc_request.set_group_name(request.GroupName());
        for (const auto& privilege : request.Privileges()) {
            auto rpc_privilege = rpc_request.mutable_privileges()->Add();
            rpc_privilege->set_name(privilege);
        }
        rpc_request.set_type(proto::milvus::OperatePrivilegeGroupType::RemovePrivilegesFromGroup);
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::OperatePrivilegeGroupRequest, proto::common::Status>(
        pre, &MilvusConnection::OperatePrivilegeGroup, nullptr);
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

Status
MilvusClientV2Impl::getFlushState(const std::vector<int64_t>& segments, bool& flushed) {
    auto pre = [&segments](proto::milvus::GetFlushStateRequest& rpc_request) {
        for (auto id : segments) {
            rpc_request.add_segmentids(id);
        }
        return Status::OK();
    };

    auto post = [&flushed](const proto::milvus::GetFlushStateResponse& response) {
        flushed = response.flushed();
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::GetFlushStateRequest, proto::milvus::GetFlushStateResponse>(
        pre, &MilvusConnection::GetFlushState, post);
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

template <typename RequestClass>
Status
MilvusClientV2Impl::iteratorPrepare(RequestClass& request) {
    CollectionDescPtr collection_desc;
    auto status = getCollectionDesc(request.DatabaseName(), request.CollectionName(), false, collection_desc);
    if (!status.IsOk()) {
        return status;
    }
    request.SetCollectionID(collection_desc->ID());

    const auto& fields = collection_desc->Schema().Fields();
    bool pk_found = false;
    for (const auto& field : fields) {
        if (field.IsPrimaryKey()) {
            request.SetPkSchema(field);
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
