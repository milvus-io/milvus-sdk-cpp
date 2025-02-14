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

#include "MilvusClientImplV2.h"

#include <chrono>
#include <nlohmann/json.hpp>
#include <thread>

#include "TypeUtils.h"
#include "common.pb.h"
#include "milvus.pb.h"
#include "schema.pb.h"

namespace milvus {

std::shared_ptr<MilvusClientV2>
MilvusClientV2::Create() {
    return std::make_shared<MilvusClientImplV2>();
}

MilvusClientImplV2::~MilvusClientImplV2() {
    Disconnect();
}

Status
MilvusClientImplV2::Connect(const ConnectParam& param) {
    if (connection_ != nullptr) {
        connection_->Disconnect();
    }

    // TODO: check connect parameter
    connection_ = std::make_shared<MilvusConnection>();
    return connection_->Connect(param);
}

Status
MilvusClientImplV2::Disconnect() {
    if (connection_ != nullptr) {
        return connection_->Disconnect();
    }

    return Status::OK();
}

Status
MilvusClientImplV2::GetServerVersion(std::string& version) {
    auto pre = []() {
        proto::milvus::GetVersionRequest rpc_request;
        return rpc_request;
    };

    auto post = [&version](const proto::milvus::GetVersionResponse& response) { version = response.version(); };

    return apiHandler<proto::milvus::GetVersionRequest, proto::milvus::GetVersionResponse>(
        pre, &MilvusConnection::GetVersion, post);
}

Status
MilvusClientImplV2::CreateCollection(const CollectionSchema& schema) {
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
MilvusClientImplV2::HasCollection(const std::string& collection_name, bool& has) {
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
MilvusClientImplV2::DropCollection(const std::string& collection_name) {
    auto pre = [&collection_name]() {
        proto::milvus::DropCollectionRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::DropCollectionRequest, proto::common::Status>(pre,
                                                                                   &MilvusConnection::DropCollection);
}

Status
MilvusClientImplV2::ListCollections(std::vector<std::string>& results, int timeout) {
    auto pre = []() {
        proto::milvus::ShowCollectionsRequest rpc_request;
        return rpc_request;
    };

    auto post = [&results](const proto::milvus::ShowCollectionsResponse& response) {
        results.reserve(response.collection_names_size());
        for (int i = 0; i < response.collection_names_size(); i++) {
            results.emplace_back(response.collection_names(i));
        }
    };

    return apiHandler<proto::milvus::ShowCollectionsRequest, proto::milvus::ShowCollectionsResponse>(
        pre, &MilvusConnection::ShowCollections, post, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::GetLoadingProgress(const std::string& collection_name, int& progress,
                                       const std::vector<std::string>& partition_names, int timeout, bool is_refresh) {
    auto pre = [&collection_name, &partition_names]() {
        proto::milvus::GetLoadingProgressRequest request;
        request.set_collection_name(collection_name);
        for (const auto& partition_name : partition_names) {
            request.add_partition_names(partition_name);
        }
        return request;
    };

    auto post = [&progress, is_refresh](const proto::milvus::GetLoadingProgressResponse& response) {
        progress = is_refresh ? response.refresh_progress() : response.progress();
    };

    return apiHandler<proto::milvus::GetLoadingProgressRequest, proto::milvus::GetLoadingProgressResponse>(
        pre, &MilvusConnection::GetLoadingProgress, post, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::WaitForLoadingCollection(const std::string& collection_name, int timeout, bool is_refresh) {
    auto start_time = std::chrono::steady_clock::now();
    auto timeout_duration = std::chrono::seconds(timeout);

    while (true) {
        if (timeout > 0) {
            auto elapsed_time = std::chrono::steady_clock::now() - start_time;
            if (elapsed_time >= timeout_duration) {
                return Status{StatusCode::TIMEOUT, "Wait for loading collection timeout: " + collection_name};
            }
        }

        milvus::LoadState load_state;
        auto status = GetLoadState(collection_name, load_state, "", 0);
        if (load_state.GetCode() == LoadStateCode::LOADED) {
            return Status::OK();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

Status
MilvusClientImplV2::LoadCollection(const std::string& collection_name, int replica_number, bool refresh,
                                   const std::string& resource_groups, const std::vector<std::string>& load_fields,
                                   bool skip_load_dynamic_field, int timeout) {
    auto pre = [&collection_name, replica_number, refresh, resource_groups, load_fields, skip_load_dynamic_field]() {
        proto::milvus::LoadCollectionRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_replica_number(replica_number);
        rpc_request.set_refresh(refresh);
        if (!resource_groups.empty()) {
            rpc_request.add_resource_groups(resource_groups);
        }

        if (!load_fields.empty()) {
            for (const auto& field : load_fields) {
                rpc_request.add_load_fields(field);
            }
        }

        rpc_request.set_skip_load_dynamic_field(skip_load_dynamic_field);

        return rpc_request;
    };

    auto wait_for_status = [this, &collection_name](const proto::common::Status& response_status) {
        return WaitForLoadingCollection(collection_name, 0, true);
    };

    return apiHandler<proto::milvus::LoadCollectionRequest, proto::common::Status>(
        pre, &MilvusConnection::LoadCollection, wait_for_status, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::ReleaseCollection(const std::string& collection_name) {
    auto pre = [&collection_name]() {
        proto::milvus::ReleaseCollectionRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::ReleaseCollectionRequest, proto::common::Status>(
        pre, &MilvusConnection::ReleaseCollection);
}

Status
MilvusClientImplV2::DescribeCollection(const std::string& collection_name, CollectionDesc& collection_desc) {
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
MilvusClientImplV2::RenameCollection(const std::string& collection_name, const std::string& new_collection_name) {
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
MilvusClientImplV2::GetCollectionStats(const std::string& collection_name, CollectionStat& collection_stat) {
    auto validate = [&collection_name, this]() {
        Status ret;
        ret = Flush(std::vector<std::string>{collection_name});
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
MilvusClientImplV2::ShowCollections(const std::vector<std::string>& collection_names,
                                    CollectionsInfo& collections_info) {
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
MilvusClientImplV2::AlterCollectionProperties(const std::string& collection_name,
                                              const std::vector<std::pair<std::string, std::string>>& properties,
                                              int timeout) {
    auto pre = [&collection_name, &properties]() {
        proto::milvus::AlterCollectionRequest rpc_request;
        rpc_request.set_collection_name(collection_name);

        for (const auto& p : properties) {
            const auto& key = p.first;
            const auto& value = p.second;
            auto* kv_pair = rpc_request.add_properties();
            kv_pair->set_key(key);
            kv_pair->set_value(value);
        }

        return rpc_request;
    };

    return apiHandler<proto::milvus::AlterCollectionRequest, proto::common::Status>(
        pre, &MilvusConnection::AlterCollection, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::DropCollectionProperties(const std::string& collection_name,
                                             const std::vector<std::string>& delete_keys, int timeout) {
    auto pre = [&collection_name, &delete_keys]() {
        proto::milvus::AlterCollectionRequest rpc_request;
        rpc_request.set_collection_name(collection_name);

        for (const auto& key : delete_keys) {
            rpc_request.add_delete_keys(key);
        }

        return rpc_request;
    };

    return apiHandler<proto::milvus::AlterCollectionRequest, proto::common::Status>(
        pre, &MilvusConnection::AlterCollection, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::AlterCollectionField(const std::string& collection_name, const std::string& field_name,
                                         const std::vector<std::pair<std::string, std::string>>& field_params,
                                         const std::string& db_name, int timeout) {
    auto pre = [&db_name, &collection_name, &field_name, &field_params]() {
        proto::milvus::AlterCollectionFieldRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_field_name(field_name);

        for (const auto& f : field_params) {
            const auto& key = f.first;
            const auto& value = f.second;
            auto* kv_pair = rpc_request.add_properties();
            kv_pair->set_key(key);
            kv_pair->set_value(value);
        }

        if (!db_name.empty()) {
            rpc_request.set_db_name(db_name);
        }

        return rpc_request;
    };

    return apiHandler<proto::milvus::AlterCollectionFieldRequest, proto::common::Status>(
        pre, &MilvusConnection::AlterCollectionField, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::CreatePartition(const std::string& collection_name, const std::string& partition_name) {
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
MilvusClientImplV2::DropPartition(const std::string& collection_name, const std::string& partition_name) {
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
MilvusClientImplV2::ListPartitions(const std::string& collection_name, std::vector<std::string>& results, int timeout) {
    auto pre = [&collection_name]() {
        proto::milvus::ShowPartitionsRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        return rpc_request;
    };

    auto post = [&results](const proto::milvus::ShowPartitionsResponse& response) {
        results.reserve(response.partition_names_size());
        for (int i = 0; i < response.partition_names_size(); i++) {
            results.emplace_back(response.partition_names(i));
        }
    };

    return apiHandler<proto::milvus::ShowPartitionsRequest, proto::milvus::ShowPartitionsResponse>(
        pre, &MilvusConnection::ShowPartitions, post, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::HasPartition(const std::string& collection_name, const std::string& partition_name, bool& has) {
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
MilvusClientImplV2::WaitForLoadingPartitions(const std::string& collection_name,
                                             const std::vector<std::string>& partition_names, int timeout) {
    auto start_time = std::chrono::steady_clock::now();
    auto timeout_duration = std::chrono::seconds(timeout);

    while (true) {
        if (timeout > 0) {
            auto elapsed_time = std::chrono::steady_clock::now() - start_time;
            if (elapsed_time >= timeout_duration) {
                return Status{StatusCode::TIMEOUT, "Wait for loading partitions timeout: " + collection_name};
            }
        }

        milvus::LoadState load_state;
        auto status = GetLoadState(collection_name, load_state, partition_names, 0);
        if (load_state.GetCode() == LoadStateCode::LOADED) {
            return Status::OK();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

Status
MilvusClientImplV2::LoadPartitions(const std::string& collection_name, const std::vector<std::string>& partition_names,
                                   int replica_number, bool refresh, const std::vector<std::string>& resource_groups,
                                   const std::vector<std::string>& load_fields, bool skip_load_dynamic_field,
                                   int timeout) {
    auto pre = [&collection_name, &partition_names, replica_number, refresh, resource_groups, load_fields,
                skip_load_dynamic_field]() {
        proto::milvus::LoadPartitionsRequest rpc_request;
        rpc_request.set_collection_name(collection_name);

        for (const auto& partition_name : partition_names) {
            rpc_request.add_partition_names(partition_name);
        }

        rpc_request.set_replica_number(replica_number);
        rpc_request.set_refresh(refresh);

        if (!resource_groups.empty()) {
            for (const auto& group : resource_groups) {
                rpc_request.add_resource_groups(group);
            }
        }

        if (!load_fields.empty()) {
            for (const auto& field : load_fields) {
                rpc_request.add_load_fields(field);
            }
        }

        rpc_request.set_skip_load_dynamic_field(skip_load_dynamic_field);

        return rpc_request;
    };

    auto wait_for_status = [this, &collection_name, &partition_names](const proto::common::Status& response_status) {
        return WaitForLoadingPartitions(collection_name, partition_names, 0);  // Timeout set to 0 for indefinite wait
    };

    return apiHandler<proto::milvus::LoadPartitionsRequest, proto::common::Status>(
        pre, &MilvusConnection::LoadPartitions, wait_for_status, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::ReleasePartitions(const std::string& collection_name,
                                      const std::vector<std::string>& partition_names, int timeout) {
    auto pre = [&collection_name, &partition_names]() {
        proto::milvus::ReleasePartitionsRequest rpc_request;

        rpc_request.set_collection_name(collection_name);
        for (const auto& partition_name : partition_names) {
            rpc_request.add_partition_names(partition_name);
        }
        return rpc_request;
    };

    return apiHandler<proto::milvus::ReleasePartitionsRequest, proto::common::Status>(
        pre, &MilvusConnection::ReleasePartitions, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::GetPartitionStats(const std::string& collection_name, const std::string& partition_name,
                                      PartitionStat& partition_stat) {
    // do flush in validate stage if needed
    auto validate = [&collection_name, this] {
        Status ret;
        ret = Flush(std::vector<std::string>{collection_name});
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
MilvusClientImplV2::ShowPartitions(const std::string& collection_name, const std::vector<std::string>& partition_names,
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
MilvusClientImplV2::GetLoadState(const std::string& collection_name, LoadState& state,
                                 const std::string& partition_name, int timeout) {
    auto pre = [&collection_name, &partition_name]() {
        proto::milvus::GetLoadStateRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        if (!partition_name.empty()) {
            rpc_request.add_partition_names(partition_name);
        }
        return rpc_request;
    };

    auto post = [&state](const proto::milvus::GetLoadStateResponse& response) {
        state.SetCode(static_cast<LoadStateCode>(response.state()));
    };

    return apiHandler<proto::milvus::GetLoadStateRequest, proto::milvus::GetLoadStateResponse>(
        pre, &MilvusConnection::GetLoadState, post, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::GetLoadState(const std::string& collection_name, LoadState& state,
                                 const std::vector<std::string>& partition_names, int timeout) {
    auto pre = [&collection_name, &partition_names]() {
        proto::milvus::GetLoadStateRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        for (const auto& partition_name : partition_names) {
            if (!partition_name.empty()) {
                rpc_request.add_partition_names(partition_name);
            }
        }
        return rpc_request;
    };

    auto post = [&state](const proto::milvus::GetLoadStateResponse& response) {
        state.SetCode(static_cast<LoadStateCode>(response.state()));
    };

    return apiHandler<proto::milvus::GetLoadStateRequest, proto::milvus::GetLoadStateResponse>(
        pre, &MilvusConnection::GetLoadState, post, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::RefreshLoad(const std::string& collection_name, int timeout) {
    auto pre = [&collection_name]() {
        proto::milvus::LoadCollectionRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_refresh(true);
        rpc_request.set_replica_number(1);
        return rpc_request;
    };

    return apiHandler<proto::milvus::LoadCollectionRequest, proto::common::Status>(
        pre, &MilvusConnection::LoadCollection, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::CreateAlias(const std::string& collection_name, const std::string& alias) {
    auto pre = [&collection_name, &alias]() {
        proto::milvus::CreateAliasRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_alias(alias);
        return rpc_request;
    };

    return apiHandler<proto::milvus::CreateAliasRequest, proto::common::Status>(pre, &MilvusConnection::CreateAlias);
}

Status
MilvusClientImplV2::DropAlias(const std::string& alias) {
    auto pre = [&alias]() {
        proto::milvus::DropAliasRequest rpc_request;
        rpc_request.set_alias(alias);
        return rpc_request;
    };

    return apiHandler<proto::milvus::DropAliasRequest, proto::common::Status>(pre, &MilvusConnection::DropAlias);
}

Status
MilvusClientImplV2::AlterAlias(const std::string& collection_name, const std::string& alias) {
    auto pre = [&collection_name, &alias]() {
        proto::milvus::AlterAliasRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_alias(alias);
        return rpc_request;
    };

    return apiHandler<proto::milvus::AlterAliasRequest, proto::common::Status>(pre, &MilvusConnection::AlterAlias);
}

Status
MilvusClientImplV2::ListAliases(const std::string& collection_name, ListAliasesResult& result, int timeout) {
    auto pre = [&collection_name]() {
        proto::milvus::ListAliasesRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        return rpc_request;
    };

    auto post = [&result](const proto::milvus::ListAliasesResponse& response) {
        result.SetDbName(response.db_name());
        result.SetCollectionName(response.collection_name());

        std::vector<std::string> aliases;
        aliases.reserve(response.aliases_size());
        aliases.insert(aliases.end(), response.aliases().begin(), response.aliases().end());
        result.SetAliases(aliases);
    };

    return apiHandler<proto::milvus::ListAliasesRequest, proto::milvus::ListAliasesResponse>(
        pre, &MilvusConnection::ListAliases, post, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::DescribeAlias(const std::string& alias, AliasDesc& alias_desc, int timeout) {
    auto pre = [&alias]() {
        proto::milvus::DescribeAliasRequest rpc_request;
        rpc_request.set_alias(alias);
        return rpc_request;
    };

    auto post = [&alias_desc](const proto::milvus::DescribeAliasResponse& response) {
        alias_desc.SetDbName(response.db_name());
        alias_desc.SetAlias(response.alias());
        alias_desc.SetCollectionName(response.collection());
    };

    return apiHandler<proto::milvus::DescribeAliasRequest, proto::milvus::DescribeAliasResponse>(
        pre, &MilvusConnection::DescribeAlias, post, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::UsingDatabase(const std::string& db_name) {
    Disconnect();
    milvus::ConnectParam connect_param(connection_->Host(), connection_->Port());
    connect_param.SetDbName(db_name);
    if (!connection_->Username().empty()) {
        connect_param.SetUsername(connection_->Username());
    }
    if (!connection_->Password().empty()) {
        connect_param.SetPassword(connection_->Password());
    }
    if (!connection_->Token().empty()) {
        connect_param.SetToken(connection_->Token());
    }
    return Connect(connect_param);
}

Status
MilvusClientImplV2::CreateDatabase(const std::string& db_name,
                                   const std::vector<std::pair<std::string, std::string>>& properties, int timeout) {
    auto pre = [&db_name, &properties]() {
        proto::milvus::CreateDatabaseRequest rpc_request;
        rpc_request.set_db_name(db_name);

        for (const auto& p : properties) {
            const auto& key = p.first;
            const auto& value = p.second;
            auto* kv_pair = rpc_request.add_properties();
            kv_pair->set_key(key);
            kv_pair->set_value(value);
        }

        return rpc_request;
    };

    return apiHandler<proto::milvus::CreateDatabaseRequest, proto::common::Status>(
        pre, &MilvusConnection::CreateDatabase, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::DropDatabase(const std::string& db_name, int timeout) {
    auto pre = [&db_name]() {
        proto::milvus::DropDatabaseRequest rpc_request;
        rpc_request.set_db_name(db_name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::DropDatabaseRequest, proto::common::Status>(pre, &MilvusConnection::DropDatabase,
                                                                                 GrpcOpts{timeout});
}

Status
MilvusClientImplV2::ListDatabases(std::vector<std::string>& db_names, int timeout) {
    auto pre = []() {
        proto::milvus::ListDatabasesRequest rpc_request;
        return rpc_request;
    };

    auto post = [&db_names](const proto::milvus::ListDatabasesResponse& response) {
        db_names.reserve(response.db_names_size());
        for (int i = 0; i < response.db_names_size(); i++) {
            db_names.emplace_back(response.db_names(i));
        }
    };

    return apiHandler<proto::milvus::ListDatabasesRequest, proto::milvus::ListDatabasesResponse>(
        pre, &MilvusConnection::ListDatabases, post, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::DescribeDatabase(const std::string& db_name, DatabaseDesc& database_desc, int timeout) {
    auto pre = [&db_name]() {
        proto::milvus::DescribeDatabaseRequest rpc_request;
        rpc_request.set_db_name(db_name);
        return rpc_request;
    };

    auto post = [&database_desc](const proto::milvus::DescribeDatabaseResponse& response) {
        database_desc.SetDbName(response.db_name());
        database_desc.SetDbID(response.dbid());
        database_desc.SetCreatedTimestamp(response.created_timestamp());

        for (const auto& prop : response.properties()) {
            database_desc.AddProperty(prop.key(), prop.value());
        }
    };

    return apiHandler<proto::milvus::DescribeDatabaseRequest, proto::milvus::DescribeDatabaseResponse>(
        pre, &MilvusConnection::DescribeDatabase, post, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::AlterDatabaseProperties(const std::string& db_name,
                                            const std::vector<std::pair<std::string, std::string>>& properties,
                                            int timeout) {
    auto pre = [&db_name, &properties]() {
        proto::milvus::AlterDatabaseRequest rpc_request;
        rpc_request.set_db_name(db_name);

        for (const auto& p : properties) {
            const auto& key = p.first;
            const auto& value = p.second;
            auto* kv_pair = rpc_request.add_properties();
            kv_pair->set_key(key);
            kv_pair->set_value(value);
        }

        return rpc_request;
    };

    return apiHandler<proto::milvus::AlterDatabaseRequest, proto::common::Status>(pre, &MilvusConnection::AlterDatabase,
                                                                                  GrpcOpts{timeout});
}

Status
MilvusClientImplV2::DropDatabaseProperties(const std::string& db_name, const std::vector<std::string>& delete_keys,
                                           int timeout) {
    auto pre = [&db_name, &delete_keys]() {
        proto::milvus::AlterDatabaseRequest rpc_request;
        rpc_request.set_db_name(db_name);

        for (const auto& key : delete_keys) {
            rpc_request.add_delete_keys(key);
        }

        return rpc_request;
    };

    return apiHandler<proto::milvus::AlterDatabaseRequest, proto::common::Status>(pre, &MilvusConnection::AlterDatabase,
                                                                                  GrpcOpts{timeout});
}

Status
MilvusClientImplV2::WaitForCreatingIndex(const std::string& collection_name, const std::string& field_name,
                                         int timeout) {
    auto start_time = std::chrono::steady_clock::now();
    auto timeout_duration = std::chrono::seconds(timeout);
    while (true) {
        if (timeout > 0) {
            auto elapsed_time = std::chrono::steady_clock::now() - start_time;
            if (elapsed_time >= timeout_duration) {
                return Status{StatusCode::TIMEOUT,
                              "Wait for creating index timeout: " + collection_name + ":" + field_name};
            }
        }

        IndexState index_state;
        auto status = GetIndexState(collection_name, field_name, index_state);
        if (!status.IsOk()) {
            return status;
        }

        if (index_state.StateCode() == IndexStateCode::FINISHED || index_state.StateCode() == IndexStateCode::NONE) {
            return Status{StatusCode::OK, "Wait for creating index finished " + index_state.FailedReason()};
        } else if (index_state.StateCode() == IndexStateCode::FAILED) {
            return Status{StatusCode::SERVER_FAILED, "Index creation failed: " + index_state.FailedReason()};
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

Status
MilvusClientImplV2::CreateIndex(const std::string& collection_name, const IndexDesc& index_desc) {
    auto validate = [&index_desc] { return index_desc.Validate(); };

    // flush before create index
    auto flush_status = Flush(std::vector<std::string>{collection_name});
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

    auto wait_for_status = [&collection_name, &index_desc, this](const proto::common::Status&) {
        return WaitForCreatingIndex(collection_name, index_desc.FieldName(), 0);
    };

    return apiHandler<proto::milvus::CreateIndexRequest, proto::common::Status>(
        validate, pre, &MilvusConnection::CreateIndex, wait_for_status, nullptr);
}

Status
MilvusClientImplV2::DescribeIndex(const std::string& collection_name, const std::string& field_name,
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
MilvusClientImplV2::GetIndexState(const std::string& collection_name, const std::string& field_name,
                                  IndexState& state) {
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
MilvusClientImplV2::GetIndexBuildProgress(const std::string& collection_name, const std::string& field_name,
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
MilvusClientImplV2::DropIndex(const std::string& collection_name, const std::string& field_name) {
    auto pre = [&collection_name, &field_name]() {
        proto::milvus::DropIndexRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_field_name(field_name);
        return rpc_request;
    };
    return apiHandler<proto::milvus::DropIndexRequest, proto::common::Status>(pre, &MilvusConnection::DropIndex);
}

Status
MilvusClientImplV2::ListIndexes(const std::string& collection_name, std::vector<std::string>& results,
                                std::vector<std::string> field_names) {
    auto pre = [&collection_name]() {
        proto::milvus::DescribeIndexRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        return rpc_request;
    };

    auto post = [&results, field_names](const proto::milvus::DescribeIndexResponse& response) {
        auto count = response.index_descriptions_size();
        for (int i = 0; i < count; ++i) {
            auto& field_name = response.index_descriptions(i).field_name();
            auto& index_name = response.index_descriptions(i).index_name();
            if (field_names.empty() ||
                std::find(field_names.begin(), field_names.end(), field_name) != field_names.end()) {
                results.push_back(field_name);
            }
        }
    };

    return apiHandler<proto::milvus::DescribeIndexRequest, proto::milvus::DescribeIndexResponse>(
        pre, &MilvusConnection::DescribeIndex, post);
}

Status
MilvusClientImplV2::Insert(const std::string& collection_name, const std::string& partition_name,
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
MilvusClientImplV2::Upsert(const std::string& collection_name, const std::string& partition_name,
                           const std::vector<FieldDataPtr>& fields, DmlResults& results) {
    auto pre = [&collection_name, &partition_name, &fields]() {
        proto::milvus::UpsertRequest rpc_request;

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

    return apiHandler<proto::milvus::UpsertRequest, proto::milvus::MutationResult>(pre, &MilvusConnection::Upsert,
                                                                                   post);
}

Status
MilvusClientImplV2::Delete(const std::string& collection_name, const std::string& partition_name,
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
MilvusClientImplV2::HybridSearch(SearchResults& results, const std::string& collection_name,
                                 const std::vector<AnnSearchRequest>& reqs, const BaseRanker& ranker, int limit,
                                 const std::vector<std::string>& partition_names,
                                 const std::vector<std::string>& output_fields, int round_decimal,
                                 const std::vector<std::pair<std::string, std::string>>& params, int timeout) {
    auto validate = [&]() {
        if (collection_name.empty()) {
            return Status{StatusCode::INVALID_AGUMENT, "Collection name cannot be empty"};
        }

        if (reqs.empty()) {
            return Status{StatusCode::INVALID_AGUMENT, "Search requests cannot be empty"};
        }

        if (limit <= 0) {
            return Status{StatusCode::INVALID_AGUMENT, "Search limit must be positive"};
        }

        CollectionDesc collection_desc;
        auto status = DescribeCollection(collection_name, collection_desc);
        if (!status.IsOk()) {
            return status;
        }

        auto anns_fields = collection_desc.Schema().AnnsFieldNames();
        for (const auto& req : reqs) {
            const auto& field_name = req.AnnsField();
            if (anns_fields.find(field_name) == anns_fields.end()) {
                return Status{StatusCode::INVALID_AGUMENT,
                              std::string(field_name + " is not a valid anns field in collection " + collection_name)};
            }
        }

        return Status::OK();
    };

    auto pre = [&]() {
        proto::milvus::HybridSearchRequest hybrid_search_request;
        hybrid_search_request.set_collection_name(collection_name);
        for (const auto& partition_name : partition_names) {
            hybrid_search_request.add_partition_names(partition_name);
        }

        for (const auto& req : reqs) {
            proto::milvus::SearchRequest search_request;
            search_request.set_collection_name(collection_name);
            search_request.set_dsl_type(proto::common::DslType::BoolExprV1);
            if (!req.Expr().empty()) {
                search_request.set_dsl(req.Expr());
            }
            for (const auto& partition_name : partition_names) {
                search_request.add_partition_names(partition_name);
            }

            // placeholders
            proto::common::PlaceholderGroup placeholder_group;
            auto& placeholder_value = *placeholder_group.add_placeholders();
            placeholder_value.set_tag("$0");
            auto target = req.TargetVectors();
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
            search_request.set_placeholder_group(std::move(placeholder_group.SerializeAsString()));

            auto kv_pair = search_request.add_search_params();
            kv_pair->set_key("anns_field");
            kv_pair->set_value(req.AnnsField());

            kv_pair = search_request.add_search_params();
            kv_pair->set_key("topk");
            kv_pair->set_value(std::to_string(req.Limit()));

            for (const auto& param : req.Param()) {
                auto* kv_pair = search_request.add_search_params();
                kv_pair->set_key(param.first);
                kv_pair->set_value(param.second);
            }

            kv_pair = search_request.add_search_params();
            kv_pair->set_key(milvus::KeyParams());
            nlohmann::json json = nlohmann::json::object();
            kv_pair->set_value(json.dump());

            hybrid_search_request.add_requests()->CopyFrom(search_request);
        }

        auto ranker_dict = ranker.Dict();

        for (const auto& ranker_dict_item : ranker_dict.items()) {
            auto* kv_pair = hybrid_search_request.add_rank_params();
            kv_pair->set_key(ranker_dict_item.key());

            auto value = ranker_dict_item.value();
            if (value.is_object()) {
                for (auto& item : value.items()) {
                    std::string str = "{\"" + item.key() + "\":" + item.value().get<std::string>() + "}";
                    kv_pair->set_value(str);
                }
            } else if (value.is_array()) {
                kv_pair->set_value(value.dump());
            } else {
                kv_pair->set_value(value.get<std::string>());
            }
        }

        auto* kv_pair = hybrid_search_request.add_rank_params();
        kv_pair->set_key("limit");
        kv_pair->set_value(std::to_string(limit));
        kv_pair = hybrid_search_request.add_rank_params();
        kv_pair->set_key("round_decimal");
        kv_pair->set_value(std::to_string(round_decimal));
        for (const auto& field : output_fields) {
            hybrid_search_request.add_output_fields(field);
        }
        for (const auto& param : params) {
            auto* kv_pair = hybrid_search_request.add_rank_params();
            kv_pair->set_key(param.first);
            kv_pair->set_value(param.second);
        }

        return hybrid_search_request;
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

    return apiHandler<proto::milvus::HybridSearchRequest, proto::milvus::SearchResults>(
        validate, pre, &MilvusConnection::HybridSearch, nullptr, post, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::Search(const SearchArguments& arguments, SearchResults& results, int timeout) {
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

        if (!arguments.GroupByField().empty()) {
            kv_pair = rpc_request.add_search_params();
            kv_pair->set_key("group_by_field");
            kv_pair->set_value(arguments.GroupByField());
        }

        if (arguments.GroupSize() > 0) {
            kv_pair = rpc_request.add_search_params();
            kv_pair->set_key("group_size");
            kv_pair->set_value(std::to_string(arguments.GroupSize()));
        }

        if (arguments.StrictGroupSize() > 0) {
            kv_pair = rpc_request.add_search_params();
            kv_pair->set_key("strict_group_size");
            kv_pair->set_value(std::to_string(arguments.StrictGroupSize()));
        }

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
MilvusClientImplV2::Query(const QueryArguments& arguments, QueryResults& results, int timeout) {
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
MilvusClientImplV2::Get(const GetArguments& arguments, QueryResults& results, int timeout) {
    CollectionDesc collection_desc;
    auto status = DescribeCollection(arguments.CollectionName(), collection_desc);
    if (!status.IsOk()) {
        return status;
    }

    std::string expr = packPksExpr(collection_desc.Schema(), arguments.Ids());
    if (expr.empty()) {
        return {StatusCode::INVALID_AGUMENT, "Failed to convert IDs to query expression"};
    }

    auto pre = [&arguments, &expr]() {
        proto::milvus::QueryRequest rpc_request;
        rpc_request.set_collection_name(arguments.CollectionName());
        for (const auto& partition_name : arguments.PartitionNames()) {
            rpc_request.add_partition_names(partition_name);
        }

        rpc_request.set_expr(expr);
        for (const auto& field : arguments.OutputFields()) {
            rpc_request.add_output_fields(field);
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
MilvusClientImplV2::ListUsers(std::vector<std::string>& results, int timeout) {
    auto pre = []() {
        proto::milvus::ListCredUsersRequest rpc_request;
        return rpc_request;
    };

    auto post = [&results](const proto::milvus::ListCredUsersResponse& response) {
        results.clear();
        for (const auto& user : response.usernames()) {
            results.emplace_back(user);
        }
    };

    return apiHandler<proto::milvus::ListCredUsersRequest, proto::milvus::ListCredUsersResponse>(
        pre, &MilvusConnection::ListCredUsers, post, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::DescribeUser(const std::string& username, UserResult& results, int timeout) {
    auto pre = [&username]() {
        proto::milvus::SelectUserRequest rpc_request;
        auto* user_entity = rpc_request.mutable_user();
        user_entity->set_name(username);
        rpc_request.set_include_role_info(true);
        return rpc_request;
    };

    auto post = [&results, username](const proto::milvus::SelectUserResponse& response) {
        for (int i = 0; i < response.results_size(); ++i) {
            const auto& user_result = response.results(i);
            if (user_result.user().name() == username) {
                results.SetUserName(user_result.user().name());
                for (const auto& role : user_result.roles()) {
                    results.AddRole(role.name());
                }
                break;
            }
        }
    };

    return apiHandler<proto::milvus::SelectUserRequest, proto::milvus::SelectUserResponse>(
        pre, &MilvusConnection::SelectUser, post, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::CreateUser(const std::string& username, const std::string& password, int timeout) {
    auto pre = [&username, &password]() {
        proto::milvus::CreateCredentialRequest rpc_request;
        rpc_request.set_username(username);
        rpc_request.set_password(milvus::Base64Encode(password));
        return rpc_request;
    };

    return apiHandler<proto::milvus::CreateCredentialRequest, proto::common::Status>(
        pre, &MilvusConnection::CreateCredential, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::UpdatePassword(const std::string& username, const std::string& old_password,
                                   const std::string& new_password, bool reset_connection, int timeout) {
    auto pre = [&username, &old_password, &new_password]() {
        proto::milvus::UpdateCredentialRequest rpc_request;
        rpc_request.set_username(username);
        rpc_request.set_oldpassword(milvus::Base64Encode(old_password));
        rpc_request.set_newpassword(milvus::Base64Encode(new_password));
        return rpc_request;
    };

    auto post = [this, reset_connection, username, new_password](const proto::common::Status& status) {
        if (reset_connection) {
            Disconnect();
            milvus::ConnectParam connect_param(connection_->Host(), connection_->Port(), username, new_password);
            if (!connection_->DbName().empty()) {
                connect_param.SetDbName(connection_->DbName());
            }
            Connect(connect_param);
        }
        return status;
    };

    return apiHandler<proto::milvus::UpdateCredentialRequest, proto::common::Status>(
        pre, &MilvusConnection::UpdateCredential, post, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::DropUser(const std::string& username, int timeout) {
    auto pre = [&username]() {
        proto::milvus::DeleteCredentialRequest rpc_request;
        rpc_request.set_username(username);
        return rpc_request;
    };

    return apiHandler<proto::milvus::DeleteCredentialRequest, proto::common::Status>(
        pre, &MilvusConnection::DeleteCredential, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::CreateRole(const std::string& role_name, int timeout) {
    auto pre = [&role_name]() {
        proto::milvus::CreateRoleRequest rpc_request;
        proto::milvus::RoleEntity* role_entity = rpc_request.mutable_entity();
        role_entity->set_name(role_name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::CreateRoleRequest, proto::common::Status>(pre, &MilvusConnection::CreateRole,
                                                                               GrpcOpts{timeout});
}

Status
MilvusClientImplV2::DropRole(const std::string& role_name, int timeout) {
    auto pre = [&role_name]() {
        proto::milvus::DropRoleRequest rpc_request;
        rpc_request.set_role_name(role_name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::DropRoleRequest, proto::common::Status>(pre, &MilvusConnection::DropRole,
                                                                             GrpcOpts{timeout});
}

Status
MilvusClientImplV2::GrantRole(const std::string& username, const std::string& role_name, int timeout) {
    auto pre = [&username, &role_name]() {
        proto::milvus::OperateUserRoleRequest rpc_request;
        rpc_request.set_username(username);
        rpc_request.set_role_name(role_name);
        rpc_request.set_type(proto::milvus::OperateUserRoleType::AddUserToRole);
        return rpc_request;
    };

    return apiHandler<proto::milvus::OperateUserRoleRequest, proto::common::Status>(
        pre, &MilvusConnection::OperateUserRole, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::RevokeRole(const std::string& username, const std::string& role_name, int timeout) {
    auto pre = [&username, &role_name]() {
        proto::milvus::OperateUserRoleRequest rpc_request;
        rpc_request.set_username(username);
        rpc_request.set_role_name(role_name);
        rpc_request.set_type(proto::milvus::OperateUserRoleType::RemoveUserFromRole);
        return rpc_request;
    };

    return apiHandler<proto::milvus::OperateUserRoleRequest, proto::common::Status>(
        pre, &MilvusConnection::OperateUserRole, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::DescribeRole(const std::string& role_name, RoleDesc& role_desc, int timeout) {
    auto pre = [&role_name]() {
        proto::milvus::SelectGrantRequest rpc_request;
        auto* entity = rpc_request.mutable_entity();
        auto* role = entity->mutable_role();
        role->set_name(role_name);
        return rpc_request;
    };

    auto post = [&role_desc, &role_name](const proto::milvus::SelectGrantResponse& response) {
        std::vector<Privilege> privileges;
        for (const auto& entity : response.entities()) {
            if (entity.role().name() == role_name) {
                Privilege p;
                p.object_type = entity.object().name();
                p.object_name = entity.object_name();
                p.db_name = entity.db_name();
                p.role_name = entity.role().name();
                p.privilege = entity.grantor().privilege().name();
                p.grantor_name = entity.grantor().user().name();
                privileges.push_back(p);
            }
        }
        role_desc = RoleDesc(role_name, privileges);
    };

    return apiHandler<proto::milvus::SelectGrantRequest, proto::milvus::SelectGrantResponse>(
        pre, &MilvusConnection::SelectGrant, post, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::ListRoles(std::vector<std::string>& roles, int timeout) {
    auto pre = []() {
        proto::milvus::SelectRoleRequest rpc_request;
        return rpc_request;
    };

    auto post = [&roles](const proto::milvus::SelectRoleResponse& response) {
        roles.clear();
        for (const auto& result : response.results()) {
            roles.emplace_back(result.role().name());
        }
    };

    return apiHandler<proto::milvus::SelectRoleRequest, proto::milvus::SelectRoleResponse>(
        pre, &MilvusConnection::SelectRole, post, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::GrantPrivilege(const std::string& role_name, const std::string& object_type,
                                   const std::string& privilege, const std::string& object_name,
                                   const std::string& db_name, int timeout) {
    auto pre = [&role_name, &object_type, &privilege, &object_name, &db_name]() {
        proto::milvus::OperatePrivilegeRequest rpc_request;

        auto* role_entity = rpc_request.mutable_entity()->mutable_role();
        role_entity->set_name(role_name);
        auto* object_entity = rpc_request.mutable_entity()->mutable_object();
        object_entity->set_name(object_type);
        rpc_request.mutable_entity()->set_object_name(object_name);
        auto* grantor = rpc_request.mutable_entity()->mutable_grantor();
        auto* privilege_entity = grantor->mutable_privilege();
        privilege_entity->set_name(privilege);

        if (!db_name.empty()) {
            rpc_request.mutable_entity()->set_db_name(db_name);
        }

        rpc_request.set_type(proto::milvus::OperatePrivilegeType::Grant);

        return rpc_request;
    };

    return apiHandler<proto::milvus::OperatePrivilegeRequest, proto::common::Status>(
        pre, &MilvusConnection::OperatePrivilege, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::RevokePrivilege(const std::string& role_name, const std::string& object_type,
                                    const std::string& privilege, const std::string& object_name,
                                    const std::string& db_name, int timeout) {
    auto pre = [&role_name, &object_type, &privilege, &object_name, &db_name]() {
        proto::milvus::OperatePrivilegeRequest rpc_request;

        auto* role_entity = rpc_request.mutable_entity()->mutable_role();
        role_entity->set_name(role_name);
        auto* object_entity = rpc_request.mutable_entity()->mutable_object();
        object_entity->set_name(object_type);
        rpc_request.mutable_entity()->set_object_name(object_name);
        auto* grantor = rpc_request.mutable_entity()->mutable_grantor();
        auto* privilege_entity = grantor->mutable_privilege();
        privilege_entity->set_name(privilege);

        if (!db_name.empty()) {
            rpc_request.mutable_entity()->set_db_name(db_name);
        }

        rpc_request.set_type(proto::milvus::OperatePrivilegeType::Revoke);

        return rpc_request;
    };

    return apiHandler<proto::milvus::OperatePrivilegeRequest, proto::common::Status>(
        pre, &MilvusConnection::OperatePrivilege, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::CreatePrivilegeGroup(const std::string& group_name, int timeout) {
    auto pre = [&group_name]() {
        proto::milvus::CreatePrivilegeGroupRequest rpc_request;
        rpc_request.set_group_name(group_name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::CreatePrivilegeGroupRequest, proto::common::Status>(
        pre, &MilvusConnection::CreatePrivilegeGroup, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::DropPrivilegeGroup(const std::string& group_name, int timeout) {
    auto pre = [&group_name]() {
        proto::milvus::DropPrivilegeGroupRequest rpc_request;
        rpc_request.set_group_name(group_name);
        return rpc_request;
    };

    return apiHandler<proto::milvus::DropPrivilegeGroupRequest, proto::common::Status>(
        pre, &MilvusConnection::DropPrivilegeGroup, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::ListPrivilegeGroups(std::vector<PrivilegeGroupInfo>& privilege_groups, int timeout) {
    auto pre = []() {
        proto::milvus::ListPrivilegeGroupsRequest rpc_request;
        return rpc_request;
    };

    auto post = [&privilege_groups](const proto::milvus::ListPrivilegeGroupsResponse& response) {
        privilege_groups.clear();
        for (const auto& group : response.privilege_groups()) {
            PrivilegeGroupInfo group_info(group.group_name());
            for (const auto& privilege : group.privileges()) {
                group_info.AddPrivilege(privilege.name());
            }
            privilege_groups.push_back(group_info);
        }
    };

    return apiHandler<proto::milvus::ListPrivilegeGroupsRequest, proto::milvus::ListPrivilegeGroupsResponse>(
        pre, &MilvusConnection::ListPrivilegeGroups, post, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::AddPrivilegesToGroup(const std::string& group_name, const std::vector<std::string>& privileges,
                                         int timeout) {
    auto pre = [&group_name, &privileges]() {
        proto::milvus::OperatePrivilegeGroupRequest rpc_request;
        rpc_request.set_group_name(group_name);

        for (const auto& privilege : privileges) {
            auto* privilege_entity = rpc_request.add_privileges();
            privilege_entity->set_name(privilege);
        }

        rpc_request.set_type(proto::milvus::OperatePrivilegeGroupType::AddPrivilegesToGroup);
        return rpc_request;
    };

    return apiHandler<proto::milvus::OperatePrivilegeGroupRequest, proto::common::Status>(
        pre, &MilvusConnection::OperatePrivilegeGroup, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::RemovePrivilegesFromGroup(const std::string& group_name, const std::vector<std::string>& privileges,
                                              int timeout) {
    auto pre = [&group_name, &privileges]() {
        proto::milvus::OperatePrivilegeGroupRequest rpc_request;
        rpc_request.set_group_name(group_name);

        for (const auto& privilege : privileges) {
            auto* privilege_entity = rpc_request.add_privileges();
            privilege_entity->set_name(privilege);
        }

        rpc_request.set_type(proto::milvus::OperatePrivilegeGroupType::RemovePrivilegesFromGroup);
        return rpc_request;
    };

    return apiHandler<proto::milvus::OperatePrivilegeGroupRequest, proto::common::Status>(
        pre, &MilvusConnection::OperatePrivilegeGroup, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::GrantPrivilegeV2(const std::string& role_name, const std::string& privilege,
                                     const std::string& collection_name, const std::string& db_name, int timeout) {
    auto pre = [&role_name, &privilege, &collection_name, &db_name]() {
        proto::milvus::OperatePrivilegeV2Request rpc_request;

        auto* role_entity = rpc_request.mutable_role();
        role_entity->set_name(role_name);

        auto* grantor = rpc_request.mutable_grantor();
        auto* privilege_entity = grantor->mutable_privilege();
        privilege_entity->set_name(privilege);
        rpc_request.set_collection_name(collection_name);
        if (!db_name.empty()) {
            rpc_request.set_db_name(db_name);
        }

        rpc_request.set_type(proto::milvus::OperatePrivilegeType::Grant);

        return rpc_request;
    };

    return apiHandler<proto::milvus::OperatePrivilegeV2Request, proto::common::Status>(
        pre, &MilvusConnection::OperatePrivilegeV2, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::RevokePrivilegeV2(const std::string& role_name, const std::string& privilege,
                                      const std::string& collection_name, const std::string& db_name, int timeout) {
    auto pre = [&role_name, &privilege, &collection_name, &db_name]() {
        proto::milvus::OperatePrivilegeV2Request rpc_request;

        auto* role_entity = rpc_request.mutable_role();
        role_entity->set_name(role_name);

        auto* grantor = rpc_request.mutable_grantor();
        auto* privilege_entity = grantor->mutable_privilege();
        privilege_entity->set_name(privilege);
        rpc_request.set_collection_name(collection_name);
        if (!db_name.empty()) {
            rpc_request.set_db_name(db_name);
        }

        rpc_request.set_type(proto::milvus::OperatePrivilegeType::Revoke);

        return rpc_request;
    };

    return apiHandler<proto::milvus::OperatePrivilegeV2Request, proto::common::Status>(
        pre, &MilvusConnection::OperatePrivilegeV2, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::CreateResourceGroup(const std::string& resource_group, const ResourceGroupConfig& config,
                                        int timeout) {
    auto pre = [&resource_group, &config]() {
        proto::milvus::CreateResourceGroupRequest rpc_request;
        rpc_request.set_resource_group(resource_group);

        auto* rg_config = rpc_request.mutable_config();
        rg_config->mutable_requests()->set_node_num(config.GetRequestsNodeNum());
        rg_config->mutable_limits()->set_node_num(config.GetLimitsNodeNum());

        for (const auto& transfer : config.GetTransferFrom()) {
            auto* transfer_from = rg_config->add_transfer_from();
            transfer_from->set_resource_group(transfer);
        }

        for (const auto& transfer : config.GetTransferTo()) {
            auto* transfer_to = rg_config->add_transfer_to();
            transfer_to->set_resource_group(transfer);
        }

        auto* node_filter = rg_config->mutable_node_filter();
        for (const auto& label : config.GetNodeLabels()) {
            auto* kv_pair = node_filter->add_node_labels();
            kv_pair->set_key(label.first);
            kv_pair->set_value(label.second);
        }

        return rpc_request;
    };

    return apiHandler<proto::milvus::CreateResourceGroupRequest, proto::common::Status>(
        pre, &MilvusConnection::CreateResourceGroup, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::DropResourceGroup(const std::string& resource_group, int timeout) {
    auto pre = [&resource_group]() {
        proto::milvus::DropResourceGroupRequest rpc_request;
        rpc_request.set_resource_group(resource_group);
        return rpc_request;
    };

    return apiHandler<proto::milvus::DropResourceGroupRequest, proto::common::Status>(
        pre, &MilvusConnection::DropResourceGroup, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::DescribeResourceGroup(const std::string& resource_group, ResourceGroupDesc& resource_group_desc,
                                          int timeout) {
    auto pre = [&resource_group]() {
        proto::milvus::DescribeResourceGroupRequest rpc_request;
        rpc_request.set_resource_group(resource_group);
        return rpc_request;
    };

    auto post = [&resource_group_desc](const proto::milvus::DescribeResourceGroupResponse& response) {
        if (response.status().code() != 0) {
            return;
        }
        const auto& rg = response.resource_group();

        ResourceGroupConfig config;
        config.SetRequestsNodeNum(rg.config().requests().node_num());
        config.SetLimitsNodeNum(rg.config().limits().node_num());

        std::vector<std::string> transfer_from;
        for (const auto& transfer : rg.config().transfer_from()) {
            transfer_from.push_back(transfer.resource_group());
        }
        config.SetTransferFrom(transfer_from);

        std::vector<std::string> transfer_to;
        for (const auto& transfer : rg.config().transfer_to()) {
            transfer_to.push_back(transfer.resource_group());
        }
        config.SetTransferTo(transfer_to);

        std::vector<std::pair<std::string, std::string>> node_labels;
        for (const auto& label : rg.config().node_filter().node_labels()) {
            node_labels.emplace_back(label.key(), label.value());
        }
        config.SetNodeLabels(node_labels);

        std::vector<NodeInfo> nodes;
        for (const auto& node : rg.nodes()) {
            nodes.emplace_back(node.node_id(), node.address(), node.hostname());
        }

        resource_group_desc = ResourceGroupDesc(
            rg.name(), rg.capacity(), rg.num_available_node(),
            std::map<std::string, int32_t>(rg.num_loaded_replica().begin(), rg.num_loaded_replica().end()),
            std::map<std::string, int32_t>(rg.num_outgoing_node().begin(), rg.num_outgoing_node().end()),
            std::map<std::string, int32_t>(rg.num_incoming_node().begin(), rg.num_incoming_node().end()), config,
            nodes);
    };

    return apiHandler<proto::milvus::DescribeResourceGroupRequest, proto::milvus::DescribeResourceGroupResponse>(
        pre, &MilvusConnection::DescribeResourceGroup, post, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::ListResourceGroups(std::vector<std::string>& resource_groups, int timeout) {
    auto pre = []() {
        proto::milvus::ListResourceGroupsRequest rpc_request;
        return rpc_request;
    };

    auto post = [&resource_groups](const proto::milvus::ListResourceGroupsResponse& response) {
        resource_groups.clear();
        if (response.status().code() != 0) {
            return;
        }
        for (const auto& group : response.resource_groups()) {
            resource_groups.push_back(group);
        }
    };

    return apiHandler<proto::milvus::ListResourceGroupsRequest, proto::milvus::ListResourceGroupsResponse>(
        pre, &MilvusConnection::ListResourceGroups, post, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::UpdateResourceGroup(const std::string& resource_group, const ResourceGroupConfig& config,
                                        int timeout) {
    auto pre = [&resource_group, &config]() {
        proto::milvus::UpdateResourceGroupsRequest rpc_request;

        auto& config_map = *rpc_request.mutable_resource_groups();
        auto* rg_config = &config_map[resource_group];

        rg_config->mutable_requests()->set_node_num(config.GetRequestsNodeNum());
        rg_config->mutable_limits()->set_node_num(config.GetLimitsNodeNum());

        for (const auto& transfer : config.GetTransferFrom()) {
            auto* transfer_from = rg_config->add_transfer_from();
            transfer_from->set_resource_group(transfer);
        }

        for (const auto& transfer : config.GetTransferTo()) {
            auto* transfer_to = rg_config->add_transfer_to();
            transfer_to->set_resource_group(transfer);
        }

        auto* node_filter = rg_config->mutable_node_filter();
        for (const auto& label : config.GetNodeLabels()) {
            auto* kv_pair = node_filter->add_node_labels();
            kv_pair->set_key(label.first);
            kv_pair->set_value(label.second);
        }

        return rpc_request;
    };

    return apiHandler<proto::milvus::UpdateResourceGroupsRequest, proto::common::Status>(
        pre, &MilvusConnection::UpdateResourceGroups, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::WaitForFlushing(const std::map<std::string, std::vector<int64_t>>& collection_segments,
                                    int timeout) {
    auto start_time = std::chrono::steady_clock::now();
    auto timeout_duration = std::chrono::seconds(timeout);

    std::set<std::string> completed_collections;

    while (completed_collections.size() < collection_segments.size()) {
        if (timeout > 0) {
            auto elapsed_time = std::chrono::steady_clock::now() - start_time;
            if (elapsed_time >= timeout_duration) {
                return Status{StatusCode::TIMEOUT, "Wait for flushing timeout"};
            }
        }

        for (const auto& pair : collection_segments) {
            const auto& collection_name = pair.first;

            if (completed_collections.find(collection_name) != completed_collections.end()) {
                continue;
            }

            bool flushed = false;
            auto status = GetFlushState(pair.second, flushed);
            if (!status.IsOk()) {
                return status;
            }

            if (flushed) {
                completed_collections.insert(collection_name);
            }
        }

        if (completed_collections.size() < collection_segments.size()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    return Status::OK();
}

Status
MilvusClientImplV2::Flush(const std::vector<std::string>& collection_names) {
    auto pre = [&collection_names]() {
        proto::milvus::FlushRequest rpc_request;
        for (const auto& collection_name : collection_names) {
            rpc_request.add_collection_names(collection_name);
        }
        return rpc_request;
    };

    auto wait_for_status = [this](const proto::milvus::FlushResponse& response) {
        std::map<std::string, std::vector<int64_t>> collection_segments;

        for (const auto& iter : response.coll_segids()) {
            const auto& ids = iter.second.data();
            std::vector<int64_t> segids;
            segids.reserve(ids.size());
            segids.insert(segids.end(), ids.begin(), ids.end());
            collection_segments.emplace(iter.first, segids);
        }

        if (collection_segments.empty()) {
            return Status::OK();
        }

        return WaitForFlushing(collection_segments, 0);
    };

    return apiHandler<proto::milvus::FlushRequest, proto::milvus::FlushResponse>(nullptr, pre, &MilvusConnection::Flush,
                                                                                 wait_for_status, nullptr);
}

Status
MilvusClientImplV2::GetFlushState(const std::vector<int64_t>& segments, bool& flushed) {
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
MilvusClientImplV2::GetPersistentSegmentInfo(const std::string& collection_name, SegmentsInfo& segments_info) {
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
MilvusClientImplV2::GetQuerySegmentInfo(const std::string& collection_name, QuerySegmentsInfo& segments_info) {
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
MilvusClientImplV2::GetMetrics(const std::string& request, std::string& response, std::string& component_name) {
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
MilvusClientImplV2::LoadBalance(int64_t src_node, const std::vector<int64_t>& dst_nodes,
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
MilvusClientImplV2::Compact(const std::string& collection_name, int64_t& compaction_id, bool is_clustering,
                            int timeout) {
    CollectionDesc collection_desc;
    auto status = DescribeCollection(collection_name, collection_desc);
    if (!status.IsOk()) {
        return status;
    }

    auto pre = [&collection_desc, &collection_name, is_clustering]() {
        proto::milvus::ManualCompactionRequest rpc_request;
        rpc_request.set_collectionid(collection_desc.ID());
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_majorcompaction(is_clustering);
        return rpc_request;
    };

    auto post = [&compaction_id](const proto::milvus::ManualCompactionResponse& response) {
        compaction_id = response.compactionid();
    };

    return apiHandler<proto::milvus::ManualCompactionRequest, proto::milvus::ManualCompactionResponse>(
        pre, &MilvusConnection::ManualCompaction, post, GrpcOpts{timeout});
}

Status
MilvusClientImplV2::GetCompactionState(int64_t compaction_id, CompactionState& compaction_state) {
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
MilvusClientImplV2::ManualCompaction(const std::string& collection_name, uint64_t travel_timestamp,
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
MilvusClientImplV2::GetCompactionPlans(int64_t compaction_id, CompactionPlans& plans) {
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
MilvusClientImplV2::CreateCredential(const std::string& username, const std::string& password) {
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
MilvusClientImplV2::UpdateCredential(const std::string& username, const std::string& old_password,
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
MilvusClientImplV2::DeleteCredential(const std::string& username) {
    auto pre = [&username]() {
        proto::milvus::DeleteCredentialRequest rpc_request;
        rpc_request.set_username(username);
        return rpc_request;
    };

    return apiHandler<proto::milvus::DeleteCredentialRequest, proto::common::Status>(
        pre, &MilvusConnection::DeleteCredential, nullptr);
}

Status
MilvusClientImplV2::ListCredUsers(std::vector<std::string>& users) {
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
MilvusClientImplV2::WaitForStatus(const std::function<Status(Progress&)>& query_function,
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
