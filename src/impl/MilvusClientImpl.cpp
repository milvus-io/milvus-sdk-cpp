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
MilvusClientImpl::LoadCollection(const std::string& collection_name, const ProgressMonitor& progress_monitor) {
    auto pre = [&collection_name]() {
        proto::milvus::LoadCollectionRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        return rpc_request;
    };

    auto wait_for_status = [this, &collection_name, &progress_monitor](const proto::common::Status&) {
        Status ret;
        waitForStatus(
            [&collection_name, this](Progress& progress) -> Status {
                CollectionsInfo collections_info;
                auto collection_names = std::vector<std::string>{collection_name};
                auto status = ShowCollections(collection_names, collections_info);
                if (not status.IsOk()) {
                    return status;
                }
                progress.total_ = collections_info.size();
                progress.finished_ = std::count_if(
                    collections_info.begin(), collections_info.end(),
                    [](const CollectionInfo& collection_info) { return collection_info.MemoryPercentage() >= 100; });
                return status;
            },
            progress_monitor, ret);
    };

    return apiHandler<proto::milvus::LoadCollectionRequest, proto::common::Status>(
        pre, &MilvusConnection::LoadCollection, wait_for_status);
}

Status
MilvusClientImpl::ReleaseCollection(const std::string& collection_name) {
    return Status::OK();
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
        aliases.insert(aliases.end(), response.aliases().begin(), response.aliases().end());

        collection_desc.SetAlias(std::move(aliases));
        collection_desc.SetCreatedTime(response.created_timestamp());
    };

    return apiHandler<proto::milvus::DescribeCollectionRequest, proto::milvus::DescribeCollectionResponse>(
        pre, &MilvusConnection::DescribeCollection, post);
}

Status
MilvusClientImpl::GetCollectionStatistics(const std::string& collection_name, const ProgressMonitor& progress_monitor,
                                          CollectionStat& collection_stat) {
    return Status::OK();
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
        for (size_t i = 0; i < response.collection_ids_size(); i++) {
            collections_info.push_back(CollectionInfo(response.collection_names(i), response.collection_ids(i),
                                                      response.created_utc_timestamps(i),
                                                      response.inmemory_percentages(i)));
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
                                 const ProgressMonitor& progress_monitor) {
    auto pre = [&collection_name, &partition_names]() {
        proto::milvus::LoadPartitionsRequest rpc_request;

        rpc_request.set_collection_name(collection_name);
        for (const auto& partition_name : partition_names) {
            rpc_request.add_partition_names(partition_name);
        }
        return rpc_request;
    };

    auto wait_for_status = [this, &collection_name, &partition_names, &progress_monitor](const proto::common::Status&) {
        Status ret;
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

                return status;
            },
            progress_monitor, ret);
        return ret;
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
                                         const ProgressMonitor& progress_monitor, PartitionStat& partition_stat) {
    return Status::OK();
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
        for (size_t i = 0; i < count; ++i) {
            partitions_info.emplace_back(response.partition_names(i), response.partitionids(i),
                                         response.created_timestamps(i), response.inmemory_percentages(i));
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
    auto pre = [&collection_name, index_desc]() {
        proto::milvus::CreateIndexRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_field_name(index_desc.FieldName());
        return rpc_request;
    };

    auto wait_for_status = [&collection_name, &index_desc, &progress_monitor, this](const proto::common::Status&) {
        Status ret;
        waitForStatus(
            [&collection_name, &index_desc, ret, this](Progress& progress) -> Status {
                IndexState index_state;
                auto status = GetIndexState(collection_name, index_desc.FieldName(), index_state);
                if (not status.IsOk()) {
                    return status;
                }

                progress.total_ = 100;

                // if index finished, progress set to 100%
                // else if index failed, return error status
                // else if index is in progressing, continue to check
                if (index_state.StateCode() == IndexStateCode::FINISHED) {
                    progress.finished_ = 100;
                } else if (index_state.StateCode() == IndexStateCode::FAILED) {
                    return Status{StatusCode::SERVER_FAILED, "index failed:" + index_state.FailedReason()};
                }

                return status;
            },
            progress_monitor, ret);
        return ret;
    };
    return apiHandler<proto::milvus::CreateIndexRequest, proto::common::Status>(
        nullptr, pre, &MilvusConnection::CreateIndex, wait_for_status, nullptr);
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

    auto post = [&id_array](const proto::milvus::MutationResult& response) {
        id_array = CreateIDArray(response.ids());
    };

    return apiHandler<proto::milvus::InsertRequest, proto::milvus::MutationResult>(pre, &MilvusConnection::Insert,
                                                                                   post);
}

Status
MilvusClientImpl::Delete(const std::string& collection_name, const std::string& partition_name,
                         const std::string& expression, IDArray& id_array) {
    auto pre = [&collection_name, &partition_name, &expression]() {
        proto::milvus::DeleteRequest rpc_request;
        rpc_request.set_collection_name(collection_name);
        rpc_request.set_partition_name(partition_name);
        rpc_request.set_expr(expression);
        return rpc_request;
    };

    auto post = [&id_array](const proto::milvus::MutationResult& response) {
        id_array = CreateIDArray(response.ids());
    };

    return apiHandler<proto::milvus::DeleteRequest, proto::milvus::MutationResult>(pre, &MilvusConnection::Delete,
                                                                                   post);
}

Status
MilvusClientImpl::Search(const SearchArguments& arguments, SearchResults& results) {
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
            // TODO(jibin): check nprobe/ef/search_k
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
        proto::milvus::PlaceholderGroup placeholder_group;
        auto& placeholder_value = *placeholder_group.add_placeholders();
        placeholder_value.set_tag("$0");
        auto target = arguments.TargetVectors();
        if (target->Type() == DataType::BINARY_VECTOR) {
            // bins
            placeholder_value.set_type(proto::milvus::PlaceholderType::BinaryVector);
            auto& bins_vec = dynamic_cast<BinaryVecFieldData&>(*target);
            for (const auto& bins : bins_vec.Data()) {
                std::string placeholder_data(reinterpret_cast<const char*>(bins.data()), bins.size());
                placeholder_value.add_values(std::move(placeholder_data));
            }
        } else {
            // floats
            placeholder_value.set_type(proto::milvus::PlaceholderType::FloatVector);
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
        kv_pair->set_key("metric_type");
        kv_pair->set_value(std::to_string(arguments.MetricType()));

        kv_pair = rpc_request.add_search_params();
        kv_pair->set_key("round_decimal");
        kv_pair->set_value(std::to_string(arguments.RoundDecimal()));

        kv_pair = rpc_request.add_search_params();
        kv_pair->set_key("params");
        nlohmann::json json_params = {};
        for (const auto& pair : arguments.ExtraParams()) {
            json_params[pair.first] = pair.second;
        }
        kv_pair->set_value(nlohmann::to_string(json_params));

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
        std::vector<int64_t> topks(num_of_queries, result_data.top_k());
        for (int i = 0; i < result_data.topks_size(); ++i) {
            topks[i] = result_data.topks(i);
        }
        std::vector<SingleResult> single_results;
        single_results.reserve(num_of_queries);
        size_t offset{0};
        for (int64_t i = 0; i < num_of_queries; ++i) {
            std::vector<float> item_scores;
            std::vector<FieldDataPtr> item_field_data;
            auto item_topk = topks[i];
            item_scores.reserve(item_topk);
            for (int64_t j = 0; j < item_topk; ++j) {
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
        validate, pre, &MilvusConnection::Search, nullptr, post);
}

Status
MilvusClientImpl::Query(const QueryArguments& arguments, QueryResults& results) {
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
    return apiHandler<proto::milvus::QueryRequest, proto::milvus::QueryResults>(pre, &MilvusConnection::Query, post);
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
        Status ret;

        waitForStatus(
            [&](Progress& p) -> Status {
                p.total_ = segment_count;

                // call GetFlushState() to check segment state
                for (auto iter = flush_segments.begin(); iter != flush_segments.end();) {
                    bool flushed = false;
                    Status status = GetFlushState(iter->second, flushed);
                    if (not status.IsOk()) {
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
            progress_monitor, ret);
        return ret;
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
    return Status::OK();
}

Status
MilvusClientImpl::GetQuerySegmentInfo(const std::string& collection_name, QuerySegmentsInfo& segments_info) {
    return Status::OK();
}

Status
MilvusClientImpl::GetMetrics(const std::string& request, std::string& response, std::string& component_name) {
    return Status::OK();
}

Status
MilvusClientImpl::LoadBalance(int64_t src_node, const std::vector<int64_t>& dst_nodes,
                              const std::vector<int64_t>& segments) {
    return Status::OK();
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

    auto pre = [&collection_name, &travel_timestamp, &collection_desc]() {
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
            source_ids.insert(source_ids.end(), info.sources().begin(), info.sources().end());

            plans.emplace_back(CompactionPlan{std::move(source_ids), info.target()});
        }
    };

    return apiHandler<proto::milvus::GetCompactionPlansRequest, proto::milvus::GetCompactionPlansResponse>(
        pre, &MilvusConnection::GetCompactionPlans, post);
}

void
MilvusClientImpl::waitForStatus(std::function<Status(Progress&)> query_function,
                                const ProgressMonitor& progress_monitor, Status& status) {
    // no need to check
    if (progress_monitor.CheckTimeout() == 0) {
        return;
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
        status = query_function(current_progress);

        // if the internal check function failed, return error
        if (not status.IsOk()) {
            break;
        }

        // notify progress
        progress_monitor.DoProgress(current_progress);

        // if progress all done, break the circle
        if (current_progress.Done()) {
            break;
        }

        // if time to deadline, return timeout error
        if (next_wait >= final_timeout) {
            status = Status{StatusCode::TIMEOUT, "time out"};
            break;
        }
    }
}

}  // namespace milvus
