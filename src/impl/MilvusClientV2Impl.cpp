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
#include <milvus/thirdparty/nlohmann/json.hpp>
#include <set>
#include <thread>
#include <type_traits>
#include <unordered_set>

#include "MilvusClientV2SessionImpl.h"
#include "rg.pb.h"
#include "types/QueryIteratorImpl.h"
#include "types/SearchIteratorImpl.h"
#include "types/SearchIteratorV2Impl.h"
#include "utils/Constants.h"
#include "utils/DmlUtils.h"
#include "utils/DqlUtils.h"
#include "utils/FieldDataSchema.h"
#include "utils/GtsDict.h"
#include "utils/MiscUtils.h"
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
MilvusClientV2Impl::Session(const std::string& cluster_id, MilvusClientV2SessionPtr& session) {
    session.reset();
    if (cluster_id.empty()) {
        return {StatusCode::INVALID_ARGUMENT, "Cluster ID cannot be empty"};
    }

    try {
        session = std::make_shared<MilvusClientV2SessionImpl>(shared_from_this(), cluster_id);
    } catch (const std::bad_weak_ptr&) {
        return {StatusCode::UNKNOWN_ERROR, "MilvusClientV2Impl must be owned by std::shared_ptr to create a session"};
    }
    return Status::OK();
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
        return {StatusCode::INVALID_ARGUMENT, "Collection schema is null"};
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
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_shards_num(static_cast<int32_t>(request.NumShards()));
        rpc_request.set_consistency_level(ConsistencyLevelCast(request.GetConsistencyLevel()));
        if (request.NumPartitions() > 0) {
            rpc_request.set_num_partitions(request.NumPartitions());
        }

        // properties
        for (const auto& it : request.Properties()) {
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

    auto post = [this, &request, &schema](const proto::common::Status& rpc_response) {
        if (request.Indexes().empty()) {
            return Status::OK();
        }

        // if user has defined indexes, create indexes immediately after collection is created.
        // note that Sync is false since the new collection empty, no need to wait index.
        const auto& descs = request.Indexes();
        for (const auto& desc : descs) {
            auto status = createIndex(request.DatabaseName(), schema.Name(), desc, false, 0);
            if (!status.IsOk()) {
                return status;
            }
        }

        // load collection automatically
        LoadCollectionRequest load_req =
            LoadCollectionRequest()
                .WithDatabaseName(request.DatabaseName())
                .WithCollectionName(schema.Name())
                .WithSync(false);  // set sync to false since no need to wait loading progress

        return LoadCollection(load_req);
    };

    return connection_.Invoke<proto::milvus::CreateCollectionRequest, proto::common::Status>(
        validate, pre, &MilvusConnection::CreateCollection, post);
}

Status
MilvusClientV2Impl::CreateCollection(const CreateSimpleCollectionRequest& request) {
    milvus::FieldSchema pk_field =
        milvus::FieldSchema(request.PrimaryFieldName(), request.PrimaryFieldType(), "", true, request.AutoID());
    if (request.PrimaryFieldType() == DataType::VARCHAR) {
        pk_field.SetMaxLength(static_cast<uint32_t>(request.MaxLength()));
    } else if (request.PrimaryFieldType() != DataType::INT64) {
        return {StatusCode::INVALID_ARGUMENT, "Primary field type is illegal"};
    }

    milvus::FieldSchema vector_field = milvus::FieldSchema(request.VectorFieldName(), DataType::FLOAT_VECTOR);
    vector_field.SetDimension(request.Dimension());

    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->AddField(std::move(pk_field));
    collection_schema->AddField(std::move(vector_field));
    collection_schema->SetEnableDynamicField(request.EnableDynamicField());

    milvus::IndexDesc index_vector(request.VectorFieldName(), "", milvus::IndexType::AUTOINDEX, request.MetricType());

    CreateCollectionRequest actual_request = CreateCollectionRequest()
                                                 .WithCollectionName(request.CollectionName())
                                                 .WithDatabaseName(request.DatabaseName())
                                                 .WithCollectionSchema(collection_schema)
                                                 .WithConsistencyLevel(request.ConsistencyLevel())
                                                 .AddIndex(std::move(index_vector));

    return CreateCollection(actual_request);
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
            auto db_name = connection_.CurrentDbName(request.DatabaseName());
            auto collection_name = request.CollectionName();
            GtsDict::GetInstance().RemoveCollectionTs(db_name, collection_name);
            removeCollectionDesc(db_name, collection_name);
        }
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::DropCollectionRequest, proto::common::Status>(
        pre, &MilvusConnection::DropCollection, post);
}

Status
MilvusClientV2Impl::TruncateCollection(const TruncateCollectionRequest& request) {
    auto pre = [&request](proto::milvus::TruncateCollectionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        return Status::OK();
    };

    auto post = [this, &request](const proto::milvus::TruncateCollectionResponse&) {
        auto db_name = connection_.CurrentDbName(request.DatabaseName());
        auto collection_name = request.CollectionName();
        GtsDict::GetInstance().RemoveCollectionTs(db_name, collection_name);
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::TruncateCollectionRequest, proto::milvus::TruncateCollectionResponse>(
        pre, &MilvusConnection::TruncateCollection, post);
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

    // wait loading progress, check load state in interval 500ms, until the time cost exceeds request.TimeoutMs()
    // ProgressMonitor timeout unit is second, it is a history problem.
    // request.TimeoutMs() 0ms is treated as 0 second, which means "forever".
    // request.TimeoutMs() in [1, 1000] is treated as 1 second, request.
    // request.TimeoutMs() in [1001, 2000] is treated as 2 seconds, etc.
    ProgressMonitor progress_monitor = ProgressMonitor::Forever();
    if (request.TimeoutMs() > 0) {
        progress_monitor = ProgressMonitor{static_cast<uint32_t>(request.TimeoutMs() + 999) / 1000};
    }
    auto wait_for_status = [this, &request, &progress_monitor](const proto::common::Status&) {
        return ConnectionHandler::WaitForStatus(
            [&request, this](Progress& progress) -> Status {
                progress.total_ = 100;
                auto db_name = connection_.CurrentDbName(request.DatabaseName());
                std::set<std::string> partition_names;
                uint32_t loading_progress = 0;
                uint32_t refresh_progress = 0;
                auto status = connection_.GetLoadingProgress(db_name, request.CollectionName(), partition_names,
                                                             loading_progress, refresh_progress);
                if (!status.IsOk()) {
                    return status;
                }
                progress.finished_ = request.Refresh() ? refresh_progress : loading_progress;
                return Status::OK();
            },
            progress_monitor);
    };
    return connection_.Invoke<proto::milvus::LoadCollectionRequest, proto::common::Status>(
        pre, &MilvusConnection::LoadCollection, wait_for_status);
}

Status
MilvusClientV2Impl::RefreshLoad(const RefreshLoadRequest& request) {
    return refreshLoad(request);
}

Status
MilvusClientV2Impl::refreshLoad(const RefreshLoadRequest& request, uint64_t rpc_timeout_ms) {
    auto pre = [&request](proto::milvus::LoadCollectionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_refresh(true);
        return Status::OK();
    };

    if (!request.Sync()) {
        return connection_.InvokeWithRpcTimeout<proto::milvus::LoadCollectionRequest, proto::common::Status>(
            rpc_timeout_ms, pre, &MilvusConnection::LoadCollection);
    }

    ProgressMonitor progress_monitor = ProgressMonitor::Forever();
    if (request.TimeoutMs() > 0) {
        progress_monitor = ProgressMonitor{static_cast<uint32_t>(request.TimeoutMs() + 999) / 1000};
    }
    auto wait_for_status = [this, &request, &progress_monitor, rpc_timeout_ms](const proto::common::Status&) {
        return ConnectionHandler::WaitForStatus(
            [&request, this, rpc_timeout_ms](Progress& progress) -> Status {
                progress.total_ = 100;
                auto db_name = connection_.CurrentDbName(request.DatabaseName());
                std::set<std::string> partition_names;
                uint32_t loading_progress = 0;
                uint32_t refresh_progress = 0;
                auto status = connection_.GetLoadingProgress(db_name, request.CollectionName(), partition_names,
                                                             loading_progress, refresh_progress, rpc_timeout_ms);
                if (!status.IsOk()) {
                    return status;
                }
                progress.finished_ = refresh_progress;
                return Status::OK();
            },
            progress_monitor);
    };
    return connection_.InvokeWithRpcTimeout<proto::milvus::LoadCollectionRequest, proto::common::Status>(
        rpc_timeout_ms, std::function<Status(void)>{}, pre, &MilvusConnection::LoadCollection, wait_for_status,
        std::function<Status(const proto::common::Status&)>{});
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
    return describeCollection(request, response);
}

Status
MilvusClientV2Impl::describeCollection(const DescribeCollectionRequest& request, DescribeCollectionResponse& response,
                                       uint64_t rpc_timeout_ms) {
    auto pre = [&request](proto::milvus::DescribeCollectionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::DescribeCollectionResponse& rpc_response) {
        CollectionDesc collection_desc;
        auto status = ConvertDescribeCollectionResponse(rpc_response, collection_desc);
        if (!status.IsOk()) {
            return status;
        }

        response.SetDesc(std::move(collection_desc));
        return Status::OK();
    };

    return connection_
        .InvokeWithRpcTimeout<proto::milvus::DescribeCollectionRequest, proto::milvus::DescribeCollectionResponse>(
            rpc_timeout_ms, pre, &MilvusConnection::DescribeCollection, post);
}

Status
MilvusClientV2Impl::BatchDescribeCollections(const BatchDescribeCollectionsRequest& request,
                                             BatchDescribeCollectionsResponse& response) {
    auto pre = [&request](proto::milvus::BatchDescribeCollectionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        for (const auto& collection_name : request.CollectionNames()) {
            rpc_request.add_collection_name(collection_name);
        }
        for (auto collection_id : request.CollectionIDs()) {
            rpc_request.add_collectionid(collection_id);
        }
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::BatchDescribeCollectionResponse& rpc_response) {
        std::vector<CollectionDesc> descs;
        descs.reserve(rpc_response.responses_size());
        for (const auto& rpc_desc : rpc_response.responses()) {
            CollectionDesc desc;
            auto status = ConvertDescribeCollectionResponse(rpc_desc, desc);
            if (!status.IsOk()) {
                return status;
            }
            descs.emplace_back(std::move(desc));
        }

        response.SetDescs(std::move(descs));
        return Status::OK();
    };

    return connection_
        .Invoke<proto::milvus::BatchDescribeCollectionRequest, proto::milvus::BatchDescribeCollectionResponse>(
            pre, &MilvusConnection::BatchDescribeCollection, post);
}

Status
MilvusClientV2Impl::DescribeReplicas(const DescribeReplicasRequest& request, DescribeReplicasResponse& response) {
    if (request.CollectionName().empty()) {
        return {StatusCode::INVALID_ARGUMENT, "Collection name cannot be empty"};
    }

    auto pre = [&request](proto::milvus::GetReplicasRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_with_shard_nodes(true);
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::GetReplicasResponse& rpc_response) {
        std::vector<ReplicaInfo> replicas;
        replicas.reserve(rpc_response.replicas_size());
        for (const auto& rpc_replica : rpc_response.replicas()) {
            ReplicaInfo replica;
            replica.SetReplicaID(rpc_replica.replicaid());
            replica.SetCollectionID(rpc_replica.collectionid());

            std::vector<int64_t> partition_ids;
            partition_ids.reserve(rpc_replica.partition_ids_size());
            partition_ids.insert(partition_ids.end(), rpc_replica.partition_ids().begin(),
                                 rpc_replica.partition_ids().end());
            replica.SetPartitionIDs(std::move(partition_ids));

            std::vector<int64_t> node_ids;
            node_ids.reserve(rpc_replica.node_ids_size());
            node_ids.insert(node_ids.end(), rpc_replica.node_ids().begin(), rpc_replica.node_ids().end());
            replica.SetNodeIDs(std::move(node_ids));

            replica.SetResourceGroupName(rpc_replica.resource_group_name());

            std::unordered_map<std::string, int32_t> num_outbound_node;
            num_outbound_node.insert(rpc_replica.num_outbound_node().begin(), rpc_replica.num_outbound_node().end());
            replica.SetNumOutboundNode(std::move(num_outbound_node));

            std::vector<ShardReplica> shard_replicas;
            shard_replicas.reserve(rpc_replica.shard_replicas_size());
            for (const auto& rpc_shard : rpc_replica.shard_replicas()) {
                ShardReplica shard_replica;
                shard_replica.SetLeaderID(rpc_shard.leaderid());
                shard_replica.SetLeaderAddress(rpc_shard.leader_addr());
                shard_replica.SetChannelName(rpc_shard.dm_channel_name());

                std::vector<int64_t> shard_node_ids;
                shard_node_ids.reserve(rpc_shard.node_ids_size());
                shard_node_ids.insert(shard_node_ids.end(), rpc_shard.node_ids().begin(), rpc_shard.node_ids().end());
                shard_replica.SetNodeIDs(std::move(shard_node_ids));

                shard_replicas.emplace_back(std::move(shard_replica));
            }
            replica.SetShardReplicas(std::move(shard_replicas));

            replicas.emplace_back(std::move(replica));
        }

        response.SetReplicas(std::move(replicas));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::GetReplicasRequest, proto::milvus::GetReplicasResponse>(
        pre, &MilvusConnection::GetReplicas, post);
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
    return getLoadState(request, response);
}

Status
MilvusClientV2Impl::getLoadState(const GetLoadStateRequest& request, GetLoadStateResponse& response,
                                 uint64_t rpc_timeout_ms) {
    auto pre = [&request](proto::milvus::GetLoadStateRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        for (const auto& partition_name : request.PartitionNames()) {
            rpc_request.add_partition_names(partition_name);
        }
        return Status::OK();
    };

    auto post = [this, &request, &response, rpc_timeout_ms](const proto::milvus::GetLoadStateResponse& rpc_response) {
        auto state = rpc_response.state();
        response.SetState(LoadStateCast(state));

        response.SetProgress(0);
        if (state == proto::common::LoadState::LoadStateLoading) {
            uint32_t progress = 0;
            uint32_t refresh_progress = 0;
            auto status =
                connection_.GetLoadingProgress(request.DatabaseName(), request.CollectionName(),
                                               request.PartitionNames(), progress, refresh_progress, rpc_timeout_ms);
            if (!status.IsOk()) {
                return status;
            }
            response.SetProgress(progress);
        } else if (state == proto::common::LoadState::LoadStateLoaded) {
            response.SetProgress(100);
        }
        return Status::OK();
    };

    return connection_.InvokeWithRpcTimeout<proto::milvus::GetLoadStateRequest, proto::milvus::GetLoadStateResponse>(
        rpc_timeout_ms, pre, &MilvusConnection::GetLoadState, post);
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
MilvusClientV2Impl::AddCollectionField(const AddCollectionFieldRequest& request) {
    auto pre = [&request](proto::milvus::AddCollectionFieldRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());

        proto::schema::FieldSchema proto_schema;
        ConvertFieldSchema(request.Field(), proto_schema);
        std::string binary;
        proto_schema.SerializeToString(&binary);
        rpc_request.set_schema(binary);

        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::AddCollectionFieldRequest, proto::common::Status>(
        pre, &MilvusConnection::AddCollectionField);
}

Status
MilvusClientV2Impl::DropCollectionField(const DropCollectionFieldRequest& request) {
    auto validate = [&request]() {
        const bool has_field_name = !request.FieldName().empty();
        if (request.FieldID() < 0) {
            return Status{StatusCode::INVALID_ARGUMENT, "Field id must be positive."};
        }
        const bool has_field_id = request.FieldID() > 0;
        if (has_field_name == has_field_id) {
            return Status{StatusCode::INVALID_ARGUMENT, "Exactly one of field name or field id must be provided."};
        }
        return Status::OK();
    };

    auto pre = [&request](proto::milvus::AlterCollectionSchemaRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());

        auto* drop_request = rpc_request.mutable_action()->mutable_drop_request();
        if (!request.FieldName().empty()) {
            drop_request->set_field_name(request.FieldName());
        } else {
            drop_request->set_field_id(request.FieldID());
        }
        return Status::OK();
    };

    return connection_
        .Invoke<proto::milvus::AlterCollectionSchemaRequest, proto::milvus::AlterCollectionSchemaResponse>(
            validate, pre, &MilvusConnection::AlterCollectionSchema);
}

Status
MilvusClientV2Impl::AddCollectionStructField(const AddCollectionStructFieldRequest& request) {
    auto validate = [&request]() {
        const auto& field = request.StructField();
        if (request.CollectionName().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "Collection name is empty"};
        }
        if (field.Name().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "Struct field name is empty"};
        }
        if (field.MaxCapacity() <= 0) {
            return Status{StatusCode::INVALID_ARGUMENT, "Struct field max capacity must be positive"};
        }
        if (field.Fields().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "Struct field must contain at least one sub field"};
        }
        if (!field.IsNullable()) {
            return Status{StatusCode::INVALID_ARGUMENT,
                          "added struct field must be nullable, please check it, struct field name = " + field.Name()};
        }
        std::set<std::string> names;
        for (const auto& sub_field : field.Fields()) {
            if (sub_field.Name().empty()) {
                return Status{StatusCode::INVALID_ARGUMENT, "Struct sub field name is empty"};
            }
            if (!names.insert(sub_field.Name()).second) {
                return Status{StatusCode::INVALID_ARGUMENT, "Duplicate struct sub field name: " + sub_field.Name()};
            }
            if (sub_field.IsPrimaryKey()) {
                return Status{StatusCode::INVALID_ARGUMENT,
                              "Struct sub field cannot be primary key: " + sub_field.Name()};
            }
            if (sub_field.IsPartitionKey()) {
                return Status{StatusCode::INVALID_ARGUMENT,
                              "Struct sub field cannot be partition key: " + sub_field.Name()};
            }
            if (sub_field.IsClusteringKey()) {
                return Status{StatusCode::INVALID_ARGUMENT,
                              "Struct sub field cannot be clustering key: " + sub_field.Name()};
            }
            if (sub_field.AutoID()) {
                return Status{StatusCode::INVALID_ARGUMENT,
                              "Struct sub field cannot enable auto id: " + sub_field.Name()};
            }
            if (sub_field.IsNullable()) {
                return Status{StatusCode::INVALID_ARGUMENT, "Struct sub field cannot be nullable: " + sub_field.Name()};
            }
            if (!sub_field.DefaultValue().is_null()) {
                return Status{StatusCode::INVALID_ARGUMENT,
                              "Struct sub field cannot have default value: " + sub_field.Name()};
            }
        }
        return Status::OK();
    };

    auto pre = [&request](proto::milvus::AddCollectionStructFieldRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        ConvertStructFieldSchema(request.StructField(), *rpc_request.mutable_struct_array_field_schema());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::AddCollectionStructFieldRequest, proto::common::Status>(
        validate, pre, &MilvusConnection::AddCollectionStructField);
}

Status
MilvusClientV2Impl::AddCollectionFunction(const AddCollectionFunctionRequest& request) {
    if (request.Function() == nullptr) {
        return {StatusCode::INVALID_ARGUMENT, "Function cannot be null."};
    }
    if (request.Function()->Name().empty()) {
        return {StatusCode::INVALID_ARGUMENT, "Function name cannot be empty."};
    }

    auto pre = [&request](proto::milvus::AddCollectionFunctionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        ConvertFunctionSchema(request.Function(), *rpc_request.mutable_functionschema());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::AddCollectionFunctionRequest, proto::common::Status>(
        pre, &MilvusConnection::AddCollectionFunction);
}

Status
MilvusClientV2Impl::AddFunctionField(const AddFunctionFieldRequest& request) {
    auto validate = [&request]() {
        if (request.Function() == nullptr) {
            return Status{StatusCode::INVALID_ARGUMENT, "Function cannot be null."};
        }
        if (request.Function()->Name().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "Function name cannot be empty."};
        }
        if (request.Field().Name().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "Field name cannot be empty."};
        }
        if (request.Function()->OutputFieldNames().size() != 1) {
            return Status{StatusCode::INVALID_ARGUMENT, "Function must have exactly one output field."};
        }
        if (request.Function()->OutputFieldNames()[0] != request.Field().Name()) {
            return Status{StatusCode::INVALID_ARGUMENT, "Function output field name must match the field being added."};
        }
        return Status::OK();
    };

    auto pre = [&request](proto::milvus::AlterCollectionSchemaRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());

        auto* add_request = rpc_request.mutable_action()->mutable_add_request();
        auto* field_info = add_request->add_field_infos();
        ConvertFieldSchema(request.Field(), *field_info->mutable_field_schema());
        field_info->mutable_field_schema()->set_is_function_output(true);
        ConvertFunctionSchema(request.Function(), *add_request->add_func_schema());
        return Status::OK();
    };

    return connection_
        .Invoke<proto::milvus::AlterCollectionSchemaRequest, proto::milvus::AlterCollectionSchemaResponse>(
            validate, pre, &MilvusConnection::AlterCollectionSchema);
}

Status
MilvusClientV2Impl::AlterCollectionFunction(const AlterCollectionFunctionRequest& request) {
    if (request.Function() == nullptr) {
        return {StatusCode::INVALID_ARGUMENT, "Function cannot be null."};
    }
    if (request.Function()->Name().empty()) {
        return {StatusCode::INVALID_ARGUMENT, "Function name cannot be empty."};
    }

    auto pre = [&request](proto::milvus::AlterCollectionFunctionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_function_name(request.Function()->Name());
        ConvertFunctionSchema(request.Function(), *rpc_request.mutable_functionschema());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::AlterCollectionFunctionRequest, proto::common::Status>(
        pre, &MilvusConnection::AlterCollectionFunction);
}

Status
MilvusClientV2Impl::DropCollectionFunction(const DropCollectionFunctionRequest& request) {
    if (request.FunctionName().empty()) {
        return {StatusCode::INVALID_ARGUMENT, "Function name cannot be empty."};
    }

    auto pre = [&request](proto::milvus::DropCollectionFunctionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_function_name(request.FunctionName());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::DropCollectionFunctionRequest, proto::common::Status>(
        pre, &MilvusConnection::DropCollectionFunction);
}

Status
MilvusClientV2Impl::DropFunctionField(const DropFunctionFieldRequest& request) {
    if (request.FunctionName().empty()) {
        return {StatusCode::INVALID_ARGUMENT, "Function name cannot be empty."};
    }

    auto pre = [&request](proto::milvus::AlterCollectionSchemaRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());

        auto* drop_request = rpc_request.mutable_action()->mutable_drop_request();
        drop_request->set_function_name(request.FunctionName());
        drop_request->set_drop_function_output_fields(true);
        return Status::OK();
    };

    return connection_
        .Invoke<proto::milvus::AlterCollectionSchemaRequest, proto::milvus::AlterCollectionSchemaResponse>(
            pre, &MilvusConnection::AlterCollectionSchema);
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

    // wait loading progress, check load state in interval 500ms, until the time cost exceeds request.TimeoutMs()
    // ProgressMonitor timeout unit is second, it is a history problem.
    // request.TimeoutMs() 0ms is treated as 0 second, which means "forever".
    // request.TimeoutMs() in [1, 1000] is treated as 1 second, request.
    // request.TimeoutMs() in [1001, 2000] is treated as 2 seconds, etc.
    ProgressMonitor progress_monitor = ProgressMonitor::Forever();
    if (request.TimeoutMs() > 0) {
        progress_monitor = ProgressMonitor{static_cast<uint32_t>(request.TimeoutMs() + 999) / 1000};
    }
    auto wait_for_status = [this, &request, &progress_monitor](const proto::common::Status&) {
        return ConnectionHandler::WaitForStatus(
            [&request, this](Progress& progress) -> Status {
                progress.total_ = 100;
                auto db_name = connection_.CurrentDbName(request.DatabaseName());
                uint32_t loading_progress = 0;
                uint32_t refresh_progress = 0;
                auto status = connection_.GetLoadingProgress(
                    db_name, request.CollectionName(), request.PartitionNames(), loading_progress, refresh_progress);
                if (!status.IsOk()) {
                    return status;
                }
                progress.finished_ = request.Refresh() ? refresh_progress : loading_progress;
                return Status::OK();
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
        db_desc.SetProperties(std::move(properties));

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
        auto status =
            createIndex(request.DatabaseName(), request.CollectionName(), desc, request.Sync(), request.TimeoutMs());
        if (!status.IsOk()) {
            return status;
        }
    }

    return Status::OK();
}

Status
MilvusClientV2Impl::DescribeIndex(const DescribeIndexRequest& request, DescribeIndexResponse& response) {
    return describeIndex(request, response);
}

Status
MilvusClientV2Impl::describeIndex(const DescribeIndexRequest& request, DescribeIndexResponse& response,
                                  uint64_t rpc_timeout_ms) {
    auto pre = [&request](proto::milvus::DescribeIndexRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        // the proto uses field_name to pass index name or field name
        if (!request.IndexName().empty()) {
            rpc_request.set_field_name(request.IndexName());
        } else if (!request.FieldName().empty()) {
            rpc_request.set_field_name(request.FieldName());
        }
        rpc_request.set_timestamp(request.Timestamp());
        return Status::OK();
    };

    auto post = [&request, &response](const proto::milvus::DescribeIndexResponse& rpc_response) {
        auto count = rpc_response.index_descriptions_size();
        if (!request.FieldName().empty() && count == 0) {
            return Status{StatusCode::SERVER_FAILED, "Index not found:" + request.FieldName()};
        }

        // althought we have specified the field_name, the server returns all the indexes of the collection,
        // pick the correct index from the list.
        const std::string& target_index_name = request.IndexName();
        const std::string& target_field_name = request.FieldName();
        std::vector<IndexDesc> descs;
        for (auto i = 0; i < count; i++) {
            auto rpc_desc = rpc_response.index_descriptions(i);

            // if field name or index name is specified, compare with the returned index description,
            // if not match, skip this index description
            // if both field name and index name are specified, both must match, otherwise skip this index description.
            // ListIndexes() inputs empty index name and field name, which means to return all indexes.
            if (!target_index_name.empty() && rpc_desc.index_name() != target_index_name) {
                continue;
            }
            if (!target_field_name.empty() && rpc_desc.field_name() != target_field_name) {
                continue;
            }

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
                } else {
                    index_desc.AddExtraParam(key, value);
                }
            }
            descs.emplace_back(std::move(index_desc));
        }

        response.SetDescs(std::move(descs));
        return Status::OK();
    };

    return connection_.InvokeWithRpcTimeout<proto::milvus::DescribeIndexRequest, proto::milvus::DescribeIndexResponse>(
        rpc_timeout_ms, pre, &MilvusConnection::DescribeIndex, post);
}

Status
MilvusClientV2Impl::ListIndexes(const ListIndexesRequest& request, ListIndexesResponse& response) {
    return listIndexes(request, response);
}

Status
MilvusClientV2Impl::listIndexes(const ListIndexesRequest& request, ListIndexesResponse& response,
                                uint64_t rpc_timeout_ms) {
    DescribeIndexRequest d_request = DescribeIndexRequest()
                                         .WithDatabaseName(request.DatabaseName())
                                         .WithCollectionName(request.CollectionName())
                                         .WithFieldName("");
    DescribeIndexResponse d_response;
    auto status = describeIndex(d_request, d_response, rpc_timeout_ms);
    if (status.IsOk()) {
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

    // if the collection has no index, the server returns an error with message like "Index not found:field_name",
    // treat it as the collection has no index, return empty list.
    if (status.ServerCode() == 700 ||
        status.LegacyServerCode() == static_cast<int32_t>(proto::common::ErrorCode::IndexNotExist)) {
        response.SetDescs({});
        response.SetIndexNames({});
        return Status::OK();
    }

    return status;
}

Status
MilvusClientV2Impl::DropIndex(const DropIndexRequest& request) {
    auto pre = [&request](proto::milvus::DropIndexRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        // the proto uses set_index_name to pass index name or field name
        if (!request.IndexName().empty()) {
            rpc_request.set_index_name(request.IndexName());
        } else if (!request.FieldName().empty()) {
            rpc_request.set_index_name(request.FieldName());
        }

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
        rpc_request.set_index_name(request.IndexName());
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
        rpc_request.set_index_name(request.IndexName());
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
            return Status{StatusCode::INVALID_ARGUMENT, "Not allow to set ColumnsData and RowsData both"};
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
            return Status{StatusCode::INVALID_ARGUMENT, "Not allow to set ColumnsData and RowsData both"};
        }

        if (!rows.empty()) {
            // verify and convert row-based data to rpc fields
            status = CheckAndSetRowData(rows, collection_desc->Schema(), request.PartialUpdate(), rpc_fields);
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
        rpc_request.set_partial_update(request.PartialUpdate());
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
            return Status{StatusCode::INVALID_ARGUMENT,
                          "Deletion condition must be specified, by primary keys or by filter expression"};
        }

        if (!request.Filter().empty() && request.IDs().GetRowCount() != 0) {
            return Status{StatusCode::INVALID_ARGUMENT,
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
    return search(request, response, "");
}

Status
MilvusClientV2Impl::search(const SearchRequest& request, SearchResponse& response, const std::string& cluster_id) {
    auto validate = [&request]() { return request.Validate(); };

    auto pre = [this, &request, &cluster_id](proto::milvus::SearchRequest& rpc_request) {
        auto current_name = connection_.CurrentDbName(request.DatabaseName());
        auto status = ConvertSearchRequest<SearchRequest>(request, current_name, rpc_request, cluster_id);
        if (!status.IsOk()) {
            return status;
        }

        if (request.Rerank()) {
            auto function_score = rpc_request.mutable_function_score();
            ConvertFunctionScore(request.Rerank(), *function_score);
        }
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
                pk_name = collection_desc->Schema().PrimaryFieldName();
            }
        }
        auto status = ConvertSearchResults(rpc_response, pk_name, results);
        response.SetResults(std::move(results));
        response.SetSessionTs(rpc_response.session_ts());
        FillSearchResponseExtraInfo(rpc_response.status(), response);
        return status;
    };

    return connection_.Invoke<proto::milvus::SearchRequest, proto::milvus::SearchResults>(
        validate, pre, &MilvusConnection::Search, nullptr, post);
}

Status
MilvusClientV2Impl::SearchIterator(SearchIteratorRequest& request, SearchIteratorPtr& iterator) {
    return searchIterator(request, iterator, "");
}

Status
MilvusClientV2Impl::searchIterator(SearchIteratorRequest& request, SearchIteratorPtr& iterator,
                                   const std::string& cluster_id) {
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
                                                                               connection_.GetRetryParam(), cluster_id);
    status = ptrV2->Init();
    iterator = ptrV2;
    if (!status.IsOk() && status.Code() == StatusCode::NOT_SUPPORTED) {
        auto ptrV1 = std::make_shared<SearchIteratorImpl<SearchIteratorRequest>>(
            connection_.GetConnection(), request, connection_.GetRetryParam(), cluster_id);
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
    return hybridSearch(request, response, "");
}

Status
MilvusClientV2Impl::hybridSearch(const HybridSearchRequest& request, HybridSearchResponse& response,
                                 const std::string& cluster_id) {
    auto pre = [this, &request, &cluster_id](proto::milvus::HybridSearchRequest& rpc_request) {
        auto current_name = connection_.CurrentDbName(request.DatabaseName());
        return ConvertHybridSearchRequest<HybridSearchRequest>(request, current_name, rpc_request, cluster_id);
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
                pk_name = collection_desc->Schema().PrimaryFieldName();
            }
        }
        auto status = ConvertSearchResults(rpc_response, pk_name, results);
        response.SetResults(std::move(results));
        response.SetSessionTs(rpc_response.session_ts());
        FillSearchResponseExtraInfo(rpc_response.status(), response);
        return status;
    };

    return connection_.Invoke<proto::milvus::HybridSearchRequest, proto::milvus::SearchResults>(
        pre, &MilvusConnection::HybridSearch, post);
}

Status
MilvusClientV2Impl::Query(const QueryRequest& request, QueryResponse& response) {
    return query(request, response, "");
}

Status
MilvusClientV2Impl::query(const QueryRequest& request, QueryResponse& response, const std::string& cluster_id) {
    auto pre = [this, &request, &cluster_id](proto::milvus::QueryRequest& rpc_request) {
        auto current_name = connection_.CurrentDbName(request.DatabaseName());
        return ConvertQueryRequest<QueryRequest>(request, current_name, rpc_request, cluster_id);
    };

    auto post = [&response](const proto::milvus::QueryResults& rpc_response) {
        QueryResults results;
        auto status = ConvertQueryResults(rpc_response, results);
        response.SetResults(std::move(results));
        response.SetSessionTs(rpc_response.session_ts());
        return status;
    };

    return connection_.Invoke<proto::milvus::QueryRequest, proto::milvus::QueryResults>(pre, &MilvusConnection::Query,
                                                                                        post);
}

Status
MilvusClientV2Impl::Get(const GetRequest& request, GetResponse& response) {
    return get(request, response, "");
}

Status
MilvusClientV2Impl::get(const GetRequest& request, GetResponse& response, const std::string& cluster_id) {
    CollectionDescPtr collection_desc;
    auto status = getCollectionDesc(request.DatabaseName(), request.CollectionName(), false, collection_desc);
    if (!status.IsOk()) {
        return status;
    }
    if (collection_desc == nullptr) {
        return {StatusCode::UNKNOWN_ERROR, "Unable to get collection schema"};
    }
    auto pk_name = collection_desc->Schema().PrimaryFieldName();

    nlohmann::json filter_template;
    const auto& id_array = request.IDs();
    if (id_array.IsIntegerID()) {
        filter_template = id_array.IntIDArray();
    } else {
        filter_template = id_array.StrIDArray();
    }

    std::set<std::string> partition_names = request.PartitionNames();  // this is a copy
    std::set<std::string> output_fields = request.OutputFields();      // this is a copy

    // use filter template to pass the id array
    static const std::string ids_key = "pks_to_get";
    auto filter = pk_name + " in {" + ids_key + "}";
    auto actual_request = QueryRequest()
                              .WithDatabaseName(request.DatabaseName())
                              .WithCollectionName(request.CollectionName())
                              .WithPartitionNames(std::move(partition_names))
                              .WithConsistencyLevel(request.GetConsistencyLevel())
                              .WithFilter(filter)
                              .AddFilterTemplate(ids_key, filter_template)
                              .WithOutputFields(std::move(output_fields));

    return query(actual_request, response, cluster_id);
}

Status
MilvusClientV2Impl::QueryIterator(QueryIteratorRequest& request, QueryIteratorPtr& iterator) {
    return queryIterator(request, iterator, "");
}

Status
MilvusClientV2Impl::queryIterator(QueryIteratorRequest& request, QueryIteratorPtr& iterator,
                                  const std::string& cluster_id) {
    auto status = iteratorPrepare(request);
    if (!status.IsOk()) {
        return status;
    }

    // iterator constructor might return error when it fails to initialize
    auto ptr = std::make_shared<QueryIteratorImpl<QueryIteratorRequest>>(connection_.GetConnection(), request,
                                                                         connection_.GetRetryParam(), cluster_id);
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

    // wait flush progress, check flush state in interval 1000ms, until the time cost exceeds request.WaitFlushedMs()
    // ProgressMonitor timeout unit is second, it is a history problem.
    // request.WaitFlushedMs() 0ms is treated as 0 second, which means "forever".
    // request.WaitFlushedMs() in [1, 1000] is treated as 1 second, request.
    // request.WaitFlushedMs() in [1001, 2000] is treated as 2 seconds, etc.
    ProgressMonitor progress_monitor = ProgressMonitor::Forever();
    progress_monitor.SetCheckInterval(1000);
    if (request.WaitFlushedMs() > 0) {
        progress_monitor = ProgressMonitor{static_cast<uint32_t>(request.WaitFlushedMs() + 999) / 1000};
        progress_monitor.SetCheckInterval(1000);
    }

    std::string db_name = request.DatabaseName();
    auto wait_for_status = [this, &progress_monitor, &db_name](const proto::milvus::FlushResponse& response) {
        std::map<std::string, std::vector<int64_t>> flush_segments;
        std::map<std::string, uint64_t> flush_tss;
        for (const auto& iter : response.coll_segids()) {
            const auto& ids = iter.second.data();
            std::vector<int64_t> seg_ids;
            seg_ids.reserve(ids.size());
            seg_ids.insert(seg_ids.end(), ids.begin(), ids.end());
            flush_segments.insert(std::make_pair(iter.first, seg_ids));
        }
        for (const auto& iter : response.coll_flush_ts()) {
            flush_tss.insert(std::make_pair(iter.first, iter.second));
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

        auto status = ConnectionHandler::WaitForStatus(
            [&segment_count, &flush_segments, &flush_tss, &finished_count, &db_name, this](Progress& p) -> Status {
                p.total_ = segment_count;

                // call GetFlushState() to check segment state
                for (auto iter = flush_segments.begin(); iter != flush_segments.end();) {
                    bool flushed = false;
                    uint64_t flush_ts = 0;
                    auto ts_iter = flush_tss.find(iter->first);
                    if (ts_iter != flush_tss.end()) {
                        flush_ts = ts_iter->second;
                    }
                    Status status = getFlushState(db_name, iter->second, flush_ts, flushed);
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

        // wait more 1 second to make sure the flushed segments are visible.
        // there might be a small delay(on server-side) between the flush_ts and the time when the segments are actually
        // flushed, if user calls createSnapshot immediately after flush returns, it might not find the flushed
        // segments.
        if (status.IsOk()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        return status;
    };

    return connection_.Invoke<proto::milvus::FlushRequest, proto::milvus::FlushResponse>(
        nullptr, pre, &MilvusConnection::Flush, wait_for_status, nullptr);
}

Status
MilvusClientV2Impl::FlushAll(const FlushAllRequest& request, FlushAllResponse& response) {
    auto pre = [&request](proto::milvus::FlushAllRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        return Status::OK();
    };

    ProgressMonitor progress_monitor = ProgressMonitor::Forever();
    if (request.WaitFlushedMs() > 0) {
        progress_monitor = ProgressMonitor{static_cast<uint32_t>(request.WaitFlushedMs() + 999) / 1000};
    }

    auto wait_for_status = [this, &request, &progress_monitor](const proto::milvus::FlushAllResponse& rpc_response) {
        GetFlushAllStateRequest state_request = GetFlushAllStateRequest()
                                                    .WithDatabaseName(request.DatabaseName())
                                                    .WithFlushAllTs(rpc_response.flush_all_ts());
        return ConnectionHandler::WaitForStatus(
            [this, &state_request](Progress& p) -> Status {
                p.total_ = 1;
                GetFlushAllStateResponse state_response;
                auto status = getFlushAllState(state_request, state_response);
                if (!status.IsOk()) {
                    return status;
                }
                p.finished_ = state_response.Flushed() ? 1 : 0;
                return Status::OK();
            },
            progress_monitor);
    };

    auto post = [&response](const proto::milvus::FlushAllResponse& rpc_response) {
        response.SetFlushAllTs(rpc_response.flush_all_ts());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::FlushAllRequest, proto::milvus::FlushAllResponse>(
        nullptr, pre, &MilvusConnection::FlushAll, wait_for_status, post);
}

Status
MilvusClientV2Impl::GetFlushAllState(const GetFlushAllStateRequest& request, GetFlushAllStateResponse& response) {
    return getFlushAllState(request, response);
}

Status
MilvusClientV2Impl::getFlushAllState(const GetFlushAllStateRequest& request, GetFlushAllStateResponse& response,
                                     uint64_t rpc_timeout_ms) {
    auto pre = [&request](proto::milvus::GetFlushAllStateRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_flush_all_ts(request.FlushAllTs());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::GetFlushAllStateResponse& rpc_response) {
        response.SetFlushed(rpc_response.flushed());
        return Status::OK();
    };

    return connection_
        .InvokeWithRpcTimeout<proto::milvus::GetFlushAllStateRequest, proto::milvus::GetFlushAllStateResponse>(
            rpc_timeout_ms, pre, &MilvusConnection::GetFlushAllState, post);
}

Status
MilvusClientV2Impl::ListPersistentSegments(const ListPersistentSegmentsRequest& request,
                                           ListPersistentSegmentsResponse& response) {
    auto pre = [&request](proto::milvus::GetPersistentSegmentInfoRequest& rpc_request) {
        rpc_request.set_dbname(request.DatabaseName());
        rpc_request.set_collectionname(request.CollectionName());
        return Status::OK();
    };

    auto post = [&request, &response](const proto::milvus::GetPersistentSegmentInfoResponse& rpc_response) {
        SegmentsInfo segments_info;
        segments_info.reserve(rpc_response.infos_size());
        for (const auto& info : rpc_response.infos()) {
            segments_info.emplace_back(info.collectionid(), info.partitionid(), info.segmentid(), info.num_rows(),
                                       SegmentStateCast(info.state()), request.CollectionName(),
                                       SegmentLevelCast(info.level()), info.storage_version(), info.is_sorted());
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

    auto post = [&request, &response](const proto::milvus::GetQuerySegmentInfoResponse& rpc_response) {
        QuerySegmentsInfo segments_info;
        segments_info.reserve(rpc_response.infos_size());
        for (const auto& info : rpc_response.infos()) {
            std::vector<int64_t> ids;
            ids.reserve(info.nodeids_size());
            for (auto id : info.nodeids()) {
                ids.push_back(id);
            }
            segments_info.emplace_back(info.collectionid(), info.partitionid(), info.segmentid(), info.num_rows(),
                                       milvus::SegmentStateCast(info.state()), info.index_name(), info.indexid(), ids,
                                       request.CollectionName(), info.mem_size(), SegmentLevelCast(info.level()),
                                       info.storage_version(), info.is_sorted());
        }
        response.SetResult(std::move(segments_info));
        return Status::OK();
    };
    return connection_.Invoke<proto::milvus::GetQuerySegmentInfoRequest, proto::milvus::GetQuerySegmentInfoResponse>(
        pre, &MilvusConnection::GetQuerySegmentInfo, post);
}

Status
MilvusClientV2Impl::Compact(const CompactRequest& request, CompactResponse& response) {
    return compact(request, response);
}

Status
MilvusClientV2Impl::compact(const CompactRequest& request, CompactResponse& response, uint64_t rpc_timeout_ms,
                            CollectionDescPtr collection_desc) {
    if (collection_desc == nullptr) {
        auto status =
            getCollectionDesc(request.DatabaseName(), request.CollectionName(), false, collection_desc, rpc_timeout_ms);
        if (!status.IsOk()) {
            return status;
        }
    }

    auto pre = [&request, &collection_desc](proto::milvus::ManualCompactionRequest& rpc_request) {
        rpc_request.set_collectionid(collection_desc->ID());
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_majorcompaction(request.ClusteringCompaction());
        if (request.TargetSize() > 0) {
            rpc_request.set_target_size(request.TargetSize());
        }
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::ManualCompactionResponse& rpc_response) {
        response.SetCompactionID(rpc_response.compactionid());
        response.SetCompactionPlanCount(rpc_response.compactionplancount());
        return Status::OK();
    };

    return connection_
        .InvokeWithRpcTimeout<proto::milvus::ManualCompactionRequest, proto::milvus::ManualCompactionResponse>(
            rpc_timeout_ms, pre, &MilvusConnection::ManualCompaction, post);
}

Status
MilvusClientV2Impl::Optimize(const OptimizeRequest& request, OptimizeTaskPtr& task) {
    task = std::make_shared<OptimizeTask>();

    if (!request.Async()) {
        OptimizeResponse response;
        auto status = runOptimize(request, *task, response);
        task->Complete(status, std::move(response));
        return status;
    }

    std::shared_ptr<MilvusClientV2Impl> self;
    try {
        self = shared_from_this();
    } catch (const std::bad_weak_ptr&) {
        return {StatusCode::UNKNOWN_ERROR, "MilvusClientV2Impl must be owned by std::shared_ptr to run Optimize"};
    }

    auto task_copy = task;
    auto status = task->Start([self, request, task_copy](OptimizeResponse& response) {
        return self->runOptimize(request, *task_copy, response);
    });
    if (!status.IsOk()) {
        return status;
    }

    return Status::OK();
}

Status
MilvusClientV2Impl::runOptimize(const OptimizeRequest& request, OptimizeTask& task, OptimizeResponse& response) {
    response.SetCollectionName(request.CollectionName());

    auto finish = [&task, &response](const Status& status) {
        if (response.StatusText().empty()) {
            response.SetStatusText(task.ShouldCancel() ? "cancelled" : (status.IsOk() ? "success" : "failed"));
        }
        return status;
    };

    auto check_cancelled = [&task]() {
        if (task.ShouldCancel()) {
            return task.CancelledStatus();
        }
        return Status::OK();
    };

    const auto start = std::chrono::steady_clock::now();
    const auto has_timeout = request.TimeoutMs() > 0;
    auto check_timeout = [&request, start, has_timeout]() {
        if (!has_timeout) {
            return Status::OK();
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        if (elapsed.count() >= request.TimeoutMs()) {
            return Status{StatusCode::TIMEOUT, "Optimize timeout"};
        }
        return Status::OK();
    };

    auto remaining_timeout_ms = [&request, start, has_timeout](int64_t& timeout_ms) {
        if (!has_timeout) {
            timeout_ms = 0;
            return Status::OK();
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        auto remaining = request.TimeoutMs() - elapsed.count();
        if (remaining <= 0) {
            return Status{StatusCode::TIMEOUT, "Optimize timeout"};
        }
        timeout_ms = remaining;
        return Status::OK();
    };

    uint64_t fallback_rpc_timeout_ms = connection_.GetRpcDeadlineMs();
    if (fallback_rpc_timeout_ms == 0) {
        fallback_rpc_timeout_ms = DEFAULT_OPTIMIZE_RPC_TIMEOUT_MS;
    }
    auto remaining_rpc_timeout_ms = [&remaining_timeout_ms, fallback_rpc_timeout_ms](uint64_t& rpc_timeout_ms) {
        int64_t timeout_ms = 0;
        auto status = remaining_timeout_ms(timeout_ms);
        if (!status.IsOk()) {
            return status;
        }
        rpc_timeout_ms = timeout_ms > 0 ? static_cast<uint64_t>(timeout_ms) : fallback_rpc_timeout_ms;
        return Status::OK();
    };

    auto check_task_active = [&check_cancelled, &check_timeout]() {
        auto status = check_cancelled();
        if (!status.IsOk()) {
            return status;
        }
        return check_timeout();
    };

    auto wait_interval = [&check_task_active]() {
        for (int i = 0; i < 10; ++i) {
            auto status = check_task_active();
            if (!status.IsOk()) {
                return status;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return Status::OK();
    };

    if (request.CollectionName().empty()) {
        return finish({StatusCode::INVALID_ARGUMENT, "Collection name cannot be empty"});
    }

    int64_t target_size_mb = 0;
    std::string normalized_target_size;
    auto status = ParseTargetSizeMB(request.TargetSize(), target_size_mb, normalized_target_size);
    if (!status.IsOk()) {
        return finish(status);
    }
    response.SetTargetSize(normalized_target_size);

    uint64_t rpc_timeout_ms = 0;
    task.AddProgress("initializing");
    CollectionDescPtr collection_desc;
    status = remaining_rpc_timeout_ms(rpc_timeout_ms);
    if (!status.IsOk()) {
        return finish(status);
    }
    status =
        getCollectionDesc(request.DatabaseName(), request.CollectionName(), false, collection_desc, rpc_timeout_ms);
    if (!status.IsOk()) {
        return finish(status);
    }

    std::unordered_set<std::string> vector_fields;
    for (const auto& field : collection_desc->Schema().Fields()) {
        if (IsVectorType(field.FieldDataType())) {
            vector_fields.insert(field.Name());
        }
    }

    std::vector<IndexDesc> vector_indexes;
    if (!vector_fields.empty()) {
        ListIndexesRequest list_request =
            ListIndexesRequest().WithDatabaseName(request.DatabaseName()).WithCollectionName(request.CollectionName());
        ListIndexesResponse list_response;
        status = remaining_rpc_timeout_ms(rpc_timeout_ms);
        if (!status.IsOk()) {
            return finish(status);
        }
        status = listIndexes(list_request, list_response, rpc_timeout_ms);
        if (!status.IsOk()) {
            return finish(status);
        }

        for (const auto& desc : list_response.Descs()) {
            if (vector_fields.count(desc.FieldName()) > 0) {
                vector_indexes.push_back(desc);
            }
        }
    }

    auto wait_vector_indexes = [this, &request, &task, &wait_interval, &remaining_rpc_timeout_ms](
                                   const std::vector<IndexDesc>& indexes, const std::string& progress) {
        if (indexes.empty()) {
            return Status::OK();
        }
        task.AddProgress(progress);

        for (;;) {
            auto all_finished = true;
            for (const auto& index : indexes) {
                DescribeIndexRequest describe_request = DescribeIndexRequest()
                                                            .WithDatabaseName(request.DatabaseName())
                                                            .WithCollectionName(request.CollectionName())
                                                            .WithFieldName(index.FieldName())
                                                            .WithIndexName(index.IndexName());
                DescribeIndexResponse describe_response;
                uint64_t rpc_timeout_ms = 0;
                auto status = remaining_rpc_timeout_ms(rpc_timeout_ms);
                if (!status.IsOk()) {
                    return status;
                }
                status = describeIndex(describe_request, describe_response, rpc_timeout_ms);
                if (!status.IsOk()) {
                    return status;
                }
                if (describe_response.Descs().empty()) {
                    return Status{StatusCode::SERVER_FAILED, "Index not found: " + index.IndexName()};
                }

                auto state = describe_response.Descs().front().StateCode();
                if (state == IndexStateCode::FAILED) {
                    return Status{StatusCode::SERVER_FAILED, describe_response.Descs().front().FailReason()};
                }
                if (state != IndexStateCode::FINISHED && state != IndexStateCode::NONE) {
                    all_finished = false;
                }
            }

            if (all_finished) {
                return Status::OK();
            }

            auto status = wait_interval();
            if (!status.IsOk()) {
                return status;
            }
        }
    };

    status = wait_vector_indexes(vector_indexes, "waiting for indexes before compaction");
    if (!status.IsOk()) {
        return finish(status);
    }
    status = check_task_active();
    if (!status.IsOk()) {
        return finish(status);
    }

    task.AddProgress("compacting");
    CompactRequest compact_request = CompactRequest()
                                         .WithDatabaseName(request.DatabaseName())
                                         .WithCollectionName(request.CollectionName())
                                         .WithTargetSize(target_size_mb);
    CompactResponse compact_response;
    status = remaining_rpc_timeout_ms(rpc_timeout_ms);
    if (!status.IsOk()) {
        return finish(status);
    }
    status = compact(compact_request, compact_response, rpc_timeout_ms, collection_desc);
    if (!status.IsOk()) {
        return finish(status);
    }
    response.SetCompactionID(compact_response.CompactionID());

    task.AddProgress("waiting for compaction");
    for (;;) {
        GetCompactionStateRequest state_request =
            GetCompactionStateRequest().WithCompactionID(compact_response.CompactionID());
        GetCompactionStateResponse state_response;
        status = remaining_rpc_timeout_ms(rpc_timeout_ms);
        if (!status.IsOk()) {
            return finish(status);
        }
        status = getCompactionState(state_request, state_response, rpc_timeout_ms);
        if (!status.IsOk()) {
            return finish(status);
        }
        if (state_response.State().FailedPlan() > 0) {
            return finish({StatusCode::SERVER_FAILED, "Compaction failed"});
        }

        auto compaction_state = state_response.State().State();
        if (compaction_state == CompactionStateCode::COMPLETED) {
            break;
        }

        status = wait_interval();
        if (!status.IsOk()) {
            return finish(status);
        }
    }

    status = wait_vector_indexes(vector_indexes, "waiting for indexes after compaction");
    if (!status.IsOk()) {
        return finish(status);
    }

    task.AddProgress("checking load state");
    GetLoadStateRequest load_state_request =
        GetLoadStateRequest().WithDatabaseName(request.DatabaseName()).WithCollectionName(request.CollectionName());
    GetLoadStateResponse load_state_response;
    status = remaining_rpc_timeout_ms(rpc_timeout_ms);
    if (!status.IsOk()) {
        return finish(status);
    }
    status = getLoadState(load_state_request, load_state_response, rpc_timeout_ms);
    if (!status.IsOk()) {
        return finish(status);
    }
    if (load_state_response.State() == LoadState::LOAD_STATE_LOADED) {
        status = check_task_active();
        if (!status.IsOk()) {
            return finish(status);
        }
        task.AddProgress("refreshing load");
        int64_t refresh_timeout_ms = 0;
        status = remaining_timeout_ms(refresh_timeout_ms);
        if (!status.IsOk()) {
            return finish(status);
        }
        status = remaining_rpc_timeout_ms(rpc_timeout_ms);
        if (!status.IsOk()) {
            return finish(status);
        }
        RefreshLoadRequest refresh_request = RefreshLoadRequest()
                                                 .WithDatabaseName(request.DatabaseName())
                                                 .WithCollectionName(request.CollectionName())
                                                 .WithSync(true)
                                                 .WithTimeoutMs(refresh_timeout_ms);
        status = refreshLoad(refresh_request, rpc_timeout_ms);
        if (!status.IsOk()) {
            return finish(status);
        }
    } else {
        task.AddProgress("collection not loaded; skip refreshLoad");
    }

    response.SetStatusText("success");
    return finish(Status::OK());
}

Status
MilvusClientV2Impl::GetCompactionState(const GetCompactionStateRequest& request, GetCompactionStateResponse& response) {
    return getCompactionState(request, response);
}

Status
MilvusClientV2Impl::getCompactionState(const GetCompactionStateRequest& request, GetCompactionStateResponse& response,
                                       uint64_t rpc_timeout_ms) {
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

    return connection_
        .InvokeWithRpcTimeout<proto::milvus::GetCompactionStateRequest, proto::milvus::GetCompactionStateResponse>(
            rpc_timeout_ms, pre, &MilvusConnection::GetCompactionState, post);
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
MilvusClientV2Impl::CreateSnapshot(const CreateSnapshotRequest& request) {
    auto validate = [&request]() {
        if (request.SnapshotName().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "Snapshot name is empty"};
        }
        if (request.CollectionName().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "Collection name is empty"};
        }
        if (request.CompactionProtectionSeconds() < 0) {
            return Status{StatusCode::INVALID_ARGUMENT, "Compaction protection seconds cannot be negative"};
        }
        return Status::OK();
    };

    auto pre = [&request](proto::milvus::CreateSnapshotRequest& rpc_request) {
        rpc_request.set_name(request.SnapshotName());
        rpc_request.set_description(request.Description());
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_compaction_protection_seconds(request.CompactionProtectionSeconds());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::CreateSnapshotRequest, proto::common::Status>(
        validate, pre, &MilvusConnection::CreateSnapshot);
}

Status
MilvusClientV2Impl::DropSnapshot(const DropSnapshotRequest& request) {
    auto validate = [&request]() {
        if (request.SnapshotName().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "Snapshot name is empty"};
        }
        if (request.CollectionName().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "Collection name is empty"};
        }
        return Status::OK();
    };

    auto pre = [&request](proto::milvus::DropSnapshotRequest& rpc_request) {
        rpc_request.set_name(request.SnapshotName());
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::DropSnapshotRequest, proto::common::Status>(
        validate, pre, &MilvusConnection::DropSnapshot);
}

Status
MilvusClientV2Impl::ListSnapshots(const ListSnapshotsRequest& request, ListSnapshotsResponse& response) {
    auto pre = [&request](proto::milvus::ListSnapshotsRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        if (!request.CollectionName().empty()) {
            rpc_request.set_collection_name(request.CollectionName());
        }
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::ListSnapshotsResponse& rpc_response) {
        std::vector<std::string> snapshots;
        snapshots.reserve(rpc_response.snapshots_size());
        snapshots.insert(snapshots.end(), rpc_response.snapshots().begin(), rpc_response.snapshots().end());
        response.SetSnapshots(std::move(snapshots));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::ListSnapshotsRequest, proto::milvus::ListSnapshotsResponse>(
        pre, &MilvusConnection::ListSnapshots, post);
}

Status
MilvusClientV2Impl::DescribeSnapshot(const DescribeSnapshotRequest& request, DescribeSnapshotResponse& response) {
    auto validate = [&request]() {
        if (request.SnapshotName().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "Snapshot name is empty"};
        }
        if (request.CollectionName().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "Collection name is empty"};
        }
        return Status::OK();
    };

    auto pre = [&request](proto::milvus::DescribeSnapshotRequest& rpc_request) {
        rpc_request.set_name(request.SnapshotName());
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::DescribeSnapshotResponse& rpc_response) {
        response.SetName(rpc_response.name());
        response.SetDescription(rpc_response.description());
        response.SetCollectionName(rpc_response.collection_name());
        std::vector<std::string> partition_names;
        partition_names.reserve(rpc_response.partition_names_size());
        partition_names.insert(partition_names.end(), rpc_response.partition_names().begin(),
                               rpc_response.partition_names().end());
        response.SetPartitionNames(std::move(partition_names));
        response.SetCreateTs(rpc_response.create_ts());
        response.SetS3Location(rpc_response.s3_location());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::DescribeSnapshotRequest, proto::milvus::DescribeSnapshotResponse>(
        validate, pre, &MilvusConnection::DescribeSnapshot, post);
}

Status
MilvusClientV2Impl::RestoreSnapshot(const RestoreSnapshotRequest& request, RestoreSnapshotResponse& response) {
    auto validate = [&request]() {
        if (request.SnapshotName().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "Snapshot name is empty"};
        }
        if (request.SourceCollectionName().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "Source collection name is empty"};
        }
        if (request.TargetCollectionName().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "Target collection name is empty"};
        }
        return Status::OK();
    };

    auto pre = [&request](proto::milvus::RestoreSnapshotRequest& rpc_request) {
        rpc_request.set_name(request.SnapshotName());
        rpc_request.set_db_name(request.SourceDatabaseName());
        rpc_request.set_collection_name(request.SourceCollectionName());
        rpc_request.set_target_db_name(request.TargetDatabaseName());
        rpc_request.set_target_collection_name(request.TargetCollectionName());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::RestoreSnapshotResponse& rpc_response) {
        response.SetJobID(rpc_response.job_id());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::RestoreSnapshotRequest, proto::milvus::RestoreSnapshotResponse>(
        validate, pre, &MilvusConnection::RestoreSnapshot, post);
}

Status
MilvusClientV2Impl::GetRestoreSnapshotState(const GetRestoreSnapshotStateRequest& request,
                                            GetRestoreSnapshotStateResponse& response) {
    auto validate = [&request]() {
        if (request.JobID() <= 0) {
            return Status{StatusCode::INVALID_ARGUMENT, "Restore snapshot job id must be positive"};
        }
        return Status::OK();
    };

    auto pre = [&request](proto::milvus::GetRestoreSnapshotStateRequest& rpc_request) {
        rpc_request.set_job_id(request.JobID());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::GetRestoreSnapshotStateResponse& rpc_response) {
        response.SetJobInfo(ConvertRestoreSnapshotJobInfo(rpc_response.info()));
        return Status::OK();
    };

    return connection_
        .Invoke<proto::milvus::GetRestoreSnapshotStateRequest, proto::milvus::GetRestoreSnapshotStateResponse>(
            validate, pre, &MilvusConnection::GetRestoreSnapshotState, post);
}

Status
MilvusClientV2Impl::ListRestoreSnapshotJobs(const ListRestoreSnapshotJobsRequest& request,
                                            ListRestoreSnapshotJobsResponse& response) {
    auto pre = [&request](proto::milvus::ListRestoreSnapshotJobsRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        if (!request.CollectionName().empty()) {
            rpc_request.set_collection_name(request.CollectionName());
        }
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::ListRestoreSnapshotJobsResponse& rpc_response) {
        std::vector<RestoreSnapshotJobInfo> jobs;
        jobs.reserve(rpc_response.jobs_size());
        for (const auto& rpc_job : rpc_response.jobs()) {
            jobs.push_back(ConvertRestoreSnapshotJobInfo(rpc_job));
        }
        response.SetJobs(std::move(jobs));
        return Status::OK();
    };

    return connection_
        .Invoke<proto::milvus::ListRestoreSnapshotJobsRequest, proto::milvus::ListRestoreSnapshotJobsResponse>(
            pre, &MilvusConnection::ListRestoreSnapshotJobs, post);
}

Status
MilvusClientV2Impl::PinSnapshotData(const PinSnapshotDataRequest& request, PinSnapshotDataResponse& response) {
    auto validate = [&request]() {
        if (request.SnapshotName().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "Snapshot name is empty"};
        }
        if (request.CollectionName().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "Collection name is empty"};
        }
        if (request.TtlSeconds() < 0) {
            return Status{StatusCode::INVALID_ARGUMENT, "TTL seconds cannot be negative"};
        }
        return Status::OK();
    };

    auto pre = [&request](proto::milvus::PinSnapshotDataRequest& rpc_request) {
        rpc_request.set_name(request.SnapshotName());
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_ttl_seconds(request.TtlSeconds());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::PinSnapshotDataResponse& rpc_response) {
        response.SetPinID(rpc_response.pin_id());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::PinSnapshotDataRequest, proto::milvus::PinSnapshotDataResponse>(
        validate, pre, &MilvusConnection::PinSnapshotData, post);
}

Status
MilvusClientV2Impl::UnpinSnapshotData(const UnpinSnapshotDataRequest& request) {
    auto validate = [&request]() {
        if (request.PinID() <= 0) {
            return Status{StatusCode::INVALID_ARGUMENT, "Snapshot pin id must be positive"};
        }
        return Status::OK();
    };

    auto pre = [&request](proto::milvus::UnpinSnapshotDataRequest& rpc_request) {
        rpc_request.set_pin_id(request.PinID());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::UnpinSnapshotDataRequest, proto::common::Status>(
        validate, pre, &MilvusConnection::UnpinSnapshotData);
}

Status
MilvusClientV2Impl::RefreshExternalCollection(const RefreshExternalCollectionRequest& request,
                                              RefreshExternalCollectionResponse& response) {
    auto validate = [&request]() {
        if (request.CollectionName().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "Collection name is empty"};
        }
        return Status::OK();
    };

    auto pre = [&request](proto::milvus::RefreshExternalCollectionRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        rpc_request.set_external_source(request.ExternalSource());
        rpc_request.set_external_spec(request.ExternalSpec().is_null() ? "" : request.ExternalSpec().dump());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::RefreshExternalCollectionResponse& rpc_response) {
        response.SetJobID(rpc_response.job_id());
        return Status::OK();
    };

    return connection_
        .Invoke<proto::milvus::RefreshExternalCollectionRequest, proto::milvus::RefreshExternalCollectionResponse>(
            validate, pre, &MilvusConnection::RefreshExternalCollection, post);
}

Status
MilvusClientV2Impl::GetRefreshExternalCollectionProgress(const GetRefreshExternalCollectionProgressRequest& request,
                                                         GetRefreshExternalCollectionProgressResponse& response) {
    auto validate = [&request]() {
        if (request.JobID() <= 0) {
            return Status{StatusCode::INVALID_ARGUMENT, "Refresh external collection job id must be positive"};
        }
        return Status::OK();
    };

    auto pre = [&request](proto::milvus::GetRefreshExternalCollectionProgressRequest& rpc_request) {
        rpc_request.set_job_id(request.JobID());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::GetRefreshExternalCollectionProgressResponse& rpc_response) {
        response.SetJobInfo(ConvertRefreshExternalCollectionJobInfo(rpc_response.job_info()));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::GetRefreshExternalCollectionProgressRequest,
                              proto::milvus::GetRefreshExternalCollectionProgressResponse>(
        validate, pre, &MilvusConnection::GetRefreshExternalCollectionProgress, post);
}

Status
MilvusClientV2Impl::ListRefreshExternalCollectionJobs(const ListRefreshExternalCollectionJobsRequest& request,
                                                      ListRefreshExternalCollectionJobsResponse& response) {
    auto pre = [&request](proto::milvus::ListRefreshExternalCollectionJobsRequest& rpc_request) {
        rpc_request.set_db_name(request.DatabaseName());
        rpc_request.set_collection_name(request.CollectionName());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::ListRefreshExternalCollectionJobsResponse& rpc_response) {
        std::vector<RefreshExternalCollectionJobInfo> jobs;
        jobs.reserve(rpc_response.jobs_size());
        for (const auto& job : rpc_response.jobs()) {
            jobs.push_back(ConvertRefreshExternalCollectionJobInfo(job));
        }
        response.SetJobs(std::move(jobs));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::ListRefreshExternalCollectionJobsRequest,
                              proto::milvus::ListRefreshExternalCollectionJobsResponse>(
        pre, &MilvusConnection::ListRefreshExternalCollectionJobs, post);
}

Status
MilvusClientV2Impl::AddFileResource(const AddFileResourceRequest& request) {
    auto validate = [&request]() {
        if (request.Name().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "File resource name is empty"};
        }
        if (request.Path().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "File resource path is empty"};
        }
        return Status::OK();
    };

    auto pre = [&request](proto::milvus::AddFileResourceRequest& rpc_request) {
        rpc_request.set_name(request.Name());
        rpc_request.set_path(request.Path());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::AddFileResourceRequest, proto::common::Status>(
        validate, pre, &MilvusConnection::AddFileResource);
}

Status
MilvusClientV2Impl::RemoveFileResource(const RemoveFileResourceRequest& request) {
    auto validate = [&request]() {
        if (request.Name().empty()) {
            return Status{StatusCode::INVALID_ARGUMENT, "File resource name is empty"};
        }
        return Status::OK();
    };

    auto pre = [&request](proto::milvus::RemoveFileResourceRequest& rpc_request) {
        rpc_request.set_name(request.Name());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::RemoveFileResourceRequest, proto::common::Status>(
        validate, pre, &MilvusConnection::RemoveFileResource);
}

Status
MilvusClientV2Impl::ListFileResources(const ListFileResourcesRequest& request, ListFileResourcesResponse& response) {
    auto post = [&response](const proto::milvus::ListFileResourcesResponse& rpc_response) {
        std::vector<FileResourceInfo> resources;
        resources.reserve(rpc_response.resources_size());
        for (const auto& resource : rpc_response.resources()) {
            resources.push_back(ConvertFileResourceInfo(resource));
        }
        response.SetResources(std::move(resources));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::ListFileResourcesRequest, proto::milvus::ListFileResourcesResponse>(
        nullptr, &MilvusConnection::ListFileResources, post);
}

Status
MilvusClientV2Impl::GetReplicateConfiguration(const GetReplicateConfigurationRequest& request,
                                              GetReplicateConfigurationResponse& response) {
    auto post = [&response](const proto::milvus::GetReplicateConfigurationResponse& rpc_response) {
        ReplicateConfiguration configuration;
        ConvertReplicateConfiguration(rpc_response.configuration(), configuration);
        response.SetConfiguration(std::move(configuration));
        return Status::OK();
    };

    return connection_
        .Invoke<proto::milvus::GetReplicateConfigurationRequest, proto::milvus::GetReplicateConfigurationResponse>(
            nullptr, &MilvusConnection::GetReplicateConfiguration, post);
}

Status
MilvusClientV2Impl::UpdateReplicateConfiguration(const UpdateReplicateConfigurationRequest& request) {
    auto pre = [&request](proto::milvus::UpdateReplicateConfigurationRequest& rpc_request) {
        ConvertReplicateConfiguration(request.Configuration(), rpc_request.mutable_replicate_configuration());
        rpc_request.set_force_promote(request.ForcePromote());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::UpdateReplicateConfigurationRequest, proto::common::Status>(
        pre, &MilvusConnection::UpdateReplicateConfiguration, nullptr);
}

Status
MilvusClientV2Impl::GetReplicateInfo(const GetReplicateInfoRequest& request, GetReplicateInfoResponse& response) {
    auto pre = [&request](proto::milvus::GetReplicateInfoRequest& rpc_request) {
        rpc_request.set_source_cluster_id(request.SourceClusterID());
        rpc_request.set_target_pchannel(request.TargetPChannel());
        return Status::OK();
    };

    auto post = [&response](const proto::milvus::GetReplicateInfoResponse& rpc_response) {
        ReplicateCheckpoint checkpoint;
        if (rpc_response.has_checkpoint()) {
            ConvertReplicateCheckpoint(rpc_response.checkpoint(), checkpoint);
        }
        response.SetCheckpoint(std::move(checkpoint));

        ReplicateCheckpoint salvage_checkpoint;
        if (rpc_response.has_salvage_checkpoint()) {
            ConvertReplicateCheckpoint(rpc_response.salvage_checkpoint(), salvage_checkpoint);
        }
        response.SetSalvageCheckpoint(std::move(salvage_checkpoint));
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::GetReplicateInfoRequest, proto::milvus::GetReplicateInfoResponse>(
        pre, &MilvusConnection::GetReplicateInfo, post);
}

Status
MilvusClientV2Impl::DumpMessages(const DumpMessagesRequest& request,
                                 const std::function<Status(const DumpedMessage&)>& on_message) {
    if (!on_message) {
        return {StatusCode::INVALID_ARGUMENT, "DumpMessages callback cannot be empty"};
    }

    proto::milvus::DumpMessagesRequest rpc_request;
    rpc_request.set_pchannel(request.PChannel());
    rpc_request.mutable_start_message_id()->set_id(request.StartMessageID().ID());
    const auto& wal_name_str = request.StartMessageID().WalName();
    if (!wal_name_str.empty()) {
        proto::common::WALName wal_name;
        if (!proto::common::WALName_Parse(wal_name_str, &wal_name)) {
            return {StatusCode::INVALID_ARGUMENT, "Unknown WAL name: " + wal_name_str};
        }
        rpc_request.mutable_start_message_id()->set_wal_name(wal_name);
    }
    rpc_request.set_start_timetick(request.StartTimeTick());
    rpc_request.set_end_timetick(request.EndTimeTick());

    auto connection = connection_.GetConnection();
    if (connection == nullptr) {
        return {StatusCode::NOT_CONNECTED, "Connection is not created!"};
    }

    auto callback = [&on_message](const proto::common::ImmutableMessage& rpc_message) {
        DumpedMessage message;
        ConvertImmutableMessage(rpc_message, message);
        return on_message(message);
    };

    return connection->DumpMessages(rpc_request, GrpcOpts{}, callback);
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
        rpc_request.set_description(request.Description());
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
MilvusClientV2Impl::UpdateUser(const UpdateUserRequest& request) {
    auto pre = [&request](proto::milvus::UpdateCredentialRequest& rpc_request) {
        rpc_request.set_username(request.UserName());
        rpc_request.set_description(request.Description());
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
            desc.SetName(result.user().name());
            desc.SetDescription(result.description());
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
        rpc_request.mutable_entity()->set_description(request.Description());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::CreateRoleRequest, proto::common::Status>(
        pre, &MilvusConnection::CreateRole, nullptr);
}

Status
MilvusClientV2Impl::AlterRole(const AlterRoleRequest& request) {
    auto pre = [&request](proto::milvus::AlterRoleRequest& rpc_request) {
        rpc_request.set_role_name(request.RoleName());
        rpc_request.set_description(request.Description());
        return Status::OK();
    };

    return connection_.Invoke<proto::milvus::AlterRoleRequest, proto::common::Status>(pre, &MilvusConnection::AlterRole,
                                                                                      nullptr);
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

    auto post = [this, &request, &response](const proto::milvus::SelectGrantResponse& rpc_response) {
        RoleDesc desc;
        desc.SetName(request.RoleName());
        for (const auto& entity : rpc_response.entities()) {
            desc.AddGrantItem({entity.object().name(), entity.object_name(), entity.db_name(), entity.role().name(),
                               entity.grantor().user().name(), entity.grantor().privilege().name()});
        }

        proto::milvus::SelectRoleRequest role_request;
        role_request.set_include_user_info(false);
        role_request.mutable_role()->set_name(request.RoleName());
        proto::milvus::SelectRoleResponse role_response;

        auto connection = connection_.GetConnection();
        if (connection == nullptr) {
            return Status{StatusCode::NOT_CONNECTED, "Connection is not created!"};
        }
        auto status = connection->SelectRole(role_request, role_response, GrpcOpts{connection_.GetRpcDeadlineMs()});
        if (!status.IsOk()) {
            return status;
        }
        if (role_response.results_size() > 0) {
            desc.SetName(role_response.results(0).role().name());
            desc.SetDescription(role_response.results(0).role().description());
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
                                bool sync, int64_t timeout_ms) {
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

    // wait index progress, check index state in interval 500ms, until the time cost exceeds timeout_ms
    // ProgressMonitor timeout unit is second, it is a history problem.
    // timeout_ms 0ms is treated as 0 second, which means "forever".
    // timeout_ms in [1, 1000] is treated as 1 second, request.
    // timeout_ms in [1001, 2000] is treated as 2 seconds, etc.
    // Note: wait timeout_ms for each index, means N indexes will wait N * timeout_ms.
    ProgressMonitor progress_monitor = ProgressMonitor::Forever();
    if (timeout_ms > 0) {
        progress_monitor = ProgressMonitor{static_cast<uint32_t>(timeout_ms + 999) / 1000};
    }
    auto wait_for_status = [&db_name, &collection_name, &desc, &progress_monitor, this](const proto::common::Status&) {
        return ConnectionHandler::WaitForStatus(
            [&db_name, &collection_name, &desc, this](Progress& progress) -> Status {
                progress.total_ = 100;

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
MilvusClientV2Impl::getFlushState(const std::string& db_name, const std::vector<int64_t>& segments, uint64_t flush_ts,
                                  bool& flushed) {
    auto actual_db = connection_.CurrentDbName(db_name);
    auto pre = [&actual_db, &segments, flush_ts](proto::milvus::GetFlushStateRequest& rpc_request) {
        rpc_request.set_db_name(actual_db);
        for (auto id : segments) {
            rpc_request.add_segmentids(id);
        }
        rpc_request.set_flush_ts(flush_ts);
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
                                      CollectionDescPtr& desc_ptr, uint64_t rpc_timeout_ms) {
    // if connection is connected to "", equals "default" db, the input db_name is "", actual_db is "default"
    // if connection is connected to "default", the input db_name is "" or "default", actual_db is "default"
    // if connection is connected to "A" but the input db_name is "B", actual_db is "B"
    // if connection is connected to "A" but the input db_name is "", actual_db is "A"
    // if connection is connected to "A" but the input db_name is "A", actual_db is "A"
    auto actual_db = connection_.CurrentDbName(db_name);

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
        DescribeCollectionRequest().WithDatabaseName(actual_db).WithCollectionName(collection_name);
    DescribeCollectionResponse response;
    auto status = describeCollection(rquest, response, rpc_timeout_ms);
    if (status.IsOk()) {
        desc_ptr = std::make_shared<CollectionDesc>(response.Desc());
        auto name = combineDbCollectionName(actual_db, collection_name);
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
    // if connection is connected to "", equals "default" db, the input db_name is "", actual_db is "default"
    // if connection is connected to "default", the input db_name is "" or "default", actual_db is "default"
    // if connection is connected to "A" but the input db_name is "B", actual_db is "B"
    // if connection is connected to "A" but the input db_name is "", actual_db is "A"
    // if connection is connected to "A" but the input db_name is "A", actual_db is "A"
    auto actual_db = connection_.CurrentDbName(db_name);

    auto name = combineDbCollectionName(actual_db, collection_name);
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
