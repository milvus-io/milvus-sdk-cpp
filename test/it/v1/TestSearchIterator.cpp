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

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <stdexcept>

#include "../mocks/MilvusMockedTest.h"
#include "../mocks/Utils.h"
#include "milvus/MilvusClientV2.h"
#include "utils/CompareUtils.h"
#include "utils/Constants.h"
#include "utils/DmlUtils.h"
#include "utils/DqlUtils.h"
#include "utils/FieldDataSchema.h"
#include "utils/TypeUtils.h"
#include "utils/cache/CollectionTsCache.h"

using ::milvus::proto::milvus::DescribeCollectionRequest;
using ::milvus::proto::milvus::DescribeCollectionResponse;
using ::milvus::proto::milvus::SearchRequest;
using ::milvus::proto::milvus::SearchResults;

using ::testing::_;
using ::testing::Property;
using ::testing::UnorderedElementsAreArray;

milvus::Status
DoSearchIterator(testing::StrictMock<milvus::MilvusMockedService>& service, milvus::MilvusClientPtr& client, bool v1,
                 milvus::ConsistencyLevel level) {
    const std::string collection_name = "Foo";
    milvus::CollectionSchema collection_schema(collection_name);
    milvus::BuildCollectionSchema(collection_schema);

    const int row_count = 20000;
    std::vector<milvus::FieldDataPtr> fields_data;
    milvus::BuildFieldsData(collection_schema, fields_data, row_count);

    std::vector<std::string> field_names;
    for (const auto& field : collection_schema.Fields()) {
        field_names.push_back(field.Name());
    }

    EXPECT_CALL(service,
                DescribeCollection(_, Property(&DescribeCollectionRequest::collection_name, collection_name), _))
        .WillOnce([&](::grpc::ServerContext*, const DescribeCollectionRequest*, DescribeCollectionResponse* response) {
            response->set_collectionid(100);
            response->set_shards_num(2);
            response->set_created_timestamp(1111);
            auto proto_schema = response->mutable_schema();
            milvus::ConvertCollectionSchema(collection_schema, *proto_schema);
            return ::grpc::Status{};
        });

    const milvus::MetricType metric = milvus::MetricType::COSINE;
    const uint64_t batch_size = 3000;
    const int64_t limit = row_count;
    uint64_t current_poz = 0;
    bool first_rpc = true;
    constexpr uint64_t probe_session_ts = 123456;
    EXPECT_CALL(service, Search(_, _, _))
        .WillRepeatedly([&](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            auto token = v1 ? "" : "dummy";
            response->mutable_results()->mutable_search_iterator_v2_results()->set_token(token);
            const bool first = first_rpc;
            first_rpc = false;
            if (first && v1) {
                return ::grpc::Status{};
            }
            if (first) {
                response->set_session_ts(probe_session_ts);
            }

            auto params = request->search_params();
            for (const auto& pair : params) {
                if (pair.key() == milvus::TOPK) {
                    EXPECT_GE(std::stoul(pair.value()), batch_size);
                }
                if (pair.key() == milvus::ITERATOR_FIELD) {
                    EXPECT_EQ(pair.value(), "True");
                }
                if (pair.key() == milvus::ITER_SEARCH_V2_KEY) {
                    EXPECT_EQ(pair.value(), "True");
                }
                if (pair.key() == milvus::ITER_SEARCH_BATCH_SIZE_KEY) {
                    EXPECT_EQ(pair.value(), std::to_string(batch_size));
                }
            }
            EXPECT_THAT(request->output_fields(), UnorderedElementsAreArray(field_names));
            EXPECT_EQ(request->collection_name(), collection_name);
            EXPECT_EQ(request->consistency_level(), milvus::ConsistencyLevelCast(level));
            if (!v1) {
                EXPECT_EQ(request->guarantee_timestamp(), first ? 0 : probe_session_ts);
            }
            response->mutable_status()->set_code(milvus::proto::common::ErrorCode::Success);
            auto* results = response->mutable_results();
            auto topk = batch_size;
            if (current_poz + batch_size > static_cast<uint64_t>(limit)) {
                topk = limit - current_poz;
            }
            results->set_top_k(topk);
            results->set_num_queries(1);
            results->set_primary_field_name(milvus::T_PK_NAME);
            auto* mutable_fields = results->mutable_fields_data();
            for (const auto& field_data : fields_data) {
                milvus::FieldDataPtr page_data;
                auto status = milvus::CopyFieldData(field_data, current_poz, current_poz + topk, page_data);
                EXPECT_TRUE(status.IsOk());
                milvus::FieldDataSchema bridge(page_data, nullptr);
                milvus::proto::schema::FieldData data;
                status = milvus::CreateProtoFieldData(bridge, data);
                EXPECT_TRUE(status.IsOk());
                mutable_fields->Add(std::move(data));

                if (field_data->Name() == milvus::T_PK_NAME) {
                    milvus::Int64FieldDataPtr ptr = std::static_pointer_cast<milvus::Int64FieldData>(field_data);
                    for (uint64_t i = 0; i < static_cast<uint64_t>(topk); i++) {
                        results->mutable_ids()->mutable_int_id()->add_data(ptr->Value(i));
                    }
                }
            }
            results->mutable_topks()->Add(topk);
            for (auto i = 0; i < topk; i++) {
                float step = (metric == milvus::MetricType::COSINE || metric == milvus::MetricType::IP) ? -0.01 : 0.01;
                results->mutable_scores()->Add(static_cast<float>(current_poz) + 100.0 + step * static_cast<float>(i));
            }
            current_poz += topk;
            return ::grpc::Status{};
        });

    milvus::SearchIteratorArguments arguments{};
    arguments.SetBatchSize(batch_size);
    arguments.SetLimit(limit);
    arguments.SetCollectionName(collection_name);
    arguments.SetFilter("id >= 0");
    arguments.SetConsistencyLevel(level);
    arguments.SetMetricType(metric);
    for (const auto& name : field_names) {
        arguments.AddOutputField(name);
    }

    std::vector<float> vector;
    vector.reserve(milvus::T_DIMENSION);
    for (auto i = 0; i < milvus::T_DIMENSION; i++) {
        vector.push_back(1.0);
    }
    auto status = arguments.AddFloat16Vector("f16_vector", vector);
    EXPECT_TRUE(status.IsOk());

    milvus::SearchIteratorPtr iterator;
    status = client->SearchIterator(arguments, iterator);
    EXPECT_TRUE(status.IsOk());

    milvus::EntityRows total_rows;
    while (true) {
        milvus::SingleResult batch_results;
        status = iterator->Next(batch_results);
        EXPECT_TRUE(status.IsOk());
        if (batch_results.GetRowCount() == 0) {
            // std::cout << "search iteration finished" << std::endl;
            break;
        }
        // std::cout << std::to_string(batch_results.GetRowCount()) + " rows fetched" << std::endl;

        milvus::EntityRows batch_rows;
        status = batch_results.OutputRows(batch_rows);
        EXPECT_TRUE(status.IsOk());
        std::copy(batch_rows.begin(), batch_rows.end(), std::back_inserter(total_rows));
    }
    EXPECT_EQ(total_rows.size(), row_count);

    milvus::SingleResult expected_results{milvus::T_PK_NAME, "score", std::move(fields_data), arguments.OutputFields()};
    milvus::EntityRows expected_rows;
    status = expected_results.OutputRows(expected_rows);
    EXPECT_TRUE(status.IsOk());

    EXPECT_EQ(total_rows.size(), expected_rows.size());
    for (auto i = 0; i < total_rows.size(); i++) {
        EXPECT_TRUE(total_rows.at(i).contains("score"));
        EXPECT_GE(total_rows.at(i)["score"], 0.0);
        total_rows.at(i).erase("score");
        EXPECT_EQ(total_rows.at(i), expected_rows.at(i));
        if (total_rows.at(i) != expected_rows.at(i)) {
            break;
        }
    }

    return milvus::Status::OK();
}

TEST_F(MilvusMockedTest, SearchIteratorV1) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    auto status = client_->Connect(connect_param);
    EXPECT_TRUE(status.IsOk());

    DoSearchIterator(service_, client_, true, milvus::ConsistencyLevel::STRONG);
}

TEST_F(MilvusMockedTest, SearchIteratorV2) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    auto status = client_->Connect(connect_param);
    EXPECT_TRUE(status.IsOk());

    DoSearchIterator(service_, client_, false, milvus::ConsistencyLevel::STRONG);
}

TEST_F(MilvusMockedTest, SearchIteratorV2BoundedFirstPageUsesServerSelectedSnapshot) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    auto status = client_->Connect(connect_param);
    EXPECT_TRUE(status.IsOk());

    DoSearchIterator(service_, client_, false, milvus::ConsistencyLevel::BOUNDED);
}

TEST_F(MilvusMockedTest, SearchIteratorV2PinsFirstBatchTimestampForSessionConsistency) {
    const std::string collection_name = "Foo";
    milvus::CollectionSchema collection_schema(collection_name);
    milvus::BuildCollectionSchema(collection_schema);

    EXPECT_CALL(service_,
                DescribeCollection(_, Property(&DescribeCollectionRequest::collection_name, collection_name), _))
        .WillOnce([&](::grpc::ServerContext*, const DescribeCollectionRequest*, DescribeCollectionResponse* response) {
            response->set_collectionid(100);
            auto proto_schema = response->mutable_schema();
            milvus::ConvertCollectionSchema(collection_schema, *proto_schema);
            return ::grpc::Status{};
        });

    const auto endpoint = "127.0.0.1:" + std::to_string(server_.ListenPort());
    constexpr uint64_t cached_dml_ts = 123456;
    constexpr uint64_t iterator_session_ts = 654321;
    milvus::CollectionTsCache::GetInstance().Set(endpoint, "default", collection_name, cached_dml_ts);

    EXPECT_CALL(service_, Search(_, _, _))
        .WillOnce([iterator_session_ts](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            EXPECT_EQ(request->guarantee_timestamp(), 0);
            response->set_session_ts(iterator_session_ts);
            auto* results = response->mutable_results();
            results->set_num_queries(1);
            results->set_top_k(1);
            results->set_primary_field_name(milvus::T_PK_NAME);
            results->mutable_topks()->Add(1);
            results->mutable_ids()->mutable_int_id()->add_data(1);
            results->mutable_scores()->Add(0.1f);
            results->mutable_search_iterator_v2_results()->set_token("dummy");
            return ::grpc::Status{};
        })
        .WillOnce([iterator_session_ts](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            EXPECT_EQ(request->guarantee_timestamp(), iterator_session_ts);
            auto* results = response->mutable_results();
            results->set_num_queries(1);
            results->set_top_k(0);
            results->set_primary_field_name(milvus::T_PK_NAME);
            results->mutable_topks()->Add(0);
            results->mutable_search_iterator_v2_results()->set_token("dummy");
            return ::grpc::Status{};
        });

    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    auto status = client_->Connect(connect_param);
    EXPECT_TRUE(status.IsOk());

    milvus::SearchIteratorArguments arguments;
    arguments.SetBatchSize(1);
    arguments.SetLimit(2);
    arguments.SetCollectionName(collection_name);
    arguments.SetConsistencyLevel(milvus::ConsistencyLevel::SESSION);
    arguments.SetMetricType(milvus::MetricType::COSINE);
    std::vector<float> vector(milvus::T_DIMENSION, 1.0f);
    status = arguments.AddFloat16Vector("f16_vector", vector);
    EXPECT_TRUE(status.IsOk());

    milvus::SearchIteratorPtr iterator;
    status = client_->SearchIterator(arguments, iterator);
    EXPECT_TRUE(status.IsOk());

    milvus::SingleResult first_page;
    status = iterator->Next(first_page);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(first_page.GetRowCount(), 1);

    milvus::SingleResult second_page;
    status = iterator->Next(second_page);
    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(second_page.GetRowCount(), 0);

    milvus::CollectionTsCache::GetInstance().Invalidate(endpoint, "default", collection_name);
}

// Verifies the client-side page filter (pymilvus external_filter_func) is applied
// on each fetched page: a fully-filtered page is dropped, and only the kept rows
// from partially-filtered pages are accumulated across pages up to the target length.
// `use_v1` forces the V1 fallback (empty token) so SearchIteratorImpl::Next runs the
// filter loop instead of SearchIteratorV2.
void
DoSearchIteratorWithExternalFilter(testing::StrictMock<milvus::MilvusMockedService>& service,
                                   milvus::MilvusClientPtr& client, bool use_v1) {
    const std::string collection_name = "Foo";
    milvus::CollectionSchema collection_schema(collection_name);
    milvus::BuildCollectionSchema(collection_schema);
    const int row_count = 10000;

    std::vector<milvus::FieldDataPtr> fields_data;
    milvus::BuildFieldsData(collection_schema, fields_data, row_count);

    std::vector<std::string> field_names;
    for (const auto& field : collection_schema.Fields()) {
        field_names.push_back(field.Name());
    }

    EXPECT_CALL(service,
                DescribeCollection(_, Property(&DescribeCollectionRequest::collection_name, collection_name), _))
        .WillOnce([&](::grpc::ServerContext*, const DescribeCollectionRequest*, DescribeCollectionResponse* response) {
            response->set_collectionid(100);
            response->set_shards_num(2);
            response->set_created_timestamp(1111);
            auto proto_schema = response->mutable_schema();
            milvus::ConvertCollectionSchema(collection_schema, *proto_schema);
            return ::grpc::Status{};
        });

    const uint64_t batch_size = 100;
    uint64_t current_poz = 0;
    bool first_rpc = true;
    EXPECT_CALL(service, Search(_, _, _))
        .WillRepeatedly([&](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            auto token = use_v1 ? "" : "dummy";
            response->mutable_results()->mutable_search_iterator_v2_results()->set_token(token);
            const bool first = first_rpc;
            first_rpc = false;
            if (first && use_v1) {
                return ::grpc::Status{};
            }
            if (first) {
                response->set_session_ts(123456);
            }

            response->mutable_status()->set_code(milvus::proto::common::ErrorCode::Success);
            auto* results = response->mutable_results();
            auto topk = batch_size;
            if (current_poz >= static_cast<uint64_t>(row_count)) {
                topk = 0;
            } else if (current_poz + batch_size > static_cast<uint64_t>(row_count)) {
                topk = row_count - current_poz;
            }
            if (topk == 0) {
                results->set_top_k(0);
                results->set_num_queries(1);
                results->set_primary_field_name(milvus::T_PK_NAME);
                results->mutable_topks()->Add(0);
                results->mutable_search_iterator_v2_results()->set_token(token);
                return ::grpc::Status{};
            }
            auto page_poz = current_poz;
            current_poz += topk;
            results->set_top_k(topk);
            results->set_num_queries(1);
            results->set_primary_field_name(milvus::T_PK_NAME);
            auto* mutable_fields = results->mutable_fields_data();
            for (const auto& field_data : fields_data) {
                // the primary key is returned via result_data.ids(), not in fields_data
                if (field_data->Name() == milvus::T_PK_NAME) {
                    continue;
                }
                milvus::FieldDataPtr page_data;
                auto cstatus = milvus::CopyFieldData(field_data, page_poz, page_poz + topk, page_data);
                EXPECT_TRUE(cstatus.IsOk());
                milvus::FieldDataSchema bridge(page_data, nullptr);
                milvus::proto::schema::FieldData data;
                cstatus = milvus::CreateProtoFieldData(bridge, data);
                EXPECT_TRUE(cstatus.IsOk());
                mutable_fields->Add(std::move(data));
            }
            milvus::Int64FieldDataPtr pk_ptr = std::static_pointer_cast<milvus::Int64FieldData>(fields_data.at(0));
            for (uint64_t i = 0; i < static_cast<uint64_t>(topk); i++) {
                results->mutable_ids()->mutable_int_id()->add_data(pk_ptr->Value(page_poz + i));
            }
            results->mutable_topks()->Add(topk);
            for (auto i = 0; i < topk; i++) {
                results->mutable_scores()->Add(static_cast<float>(page_poz) + 100.0 - 0.01f * static_cast<float>(i));
            }
            return ::grpc::Status{};
        });

    milvus::SearchIteratorArguments arguments{};
    arguments.SetBatchSize(batch_size);
    arguments.SetLimit(row_count);
    arguments.SetCollectionName(collection_name);
    arguments.SetFilter("id >= 0");
    arguments.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    arguments.SetMetricType(milvus::MetricType::COSINE);
    for (const auto& name : field_names) {
        arguments.AddOutputField(name);
    }
    std::vector<float> vector(milvus::T_DIMENSION, 1.0f);
    auto status = arguments.AddFloat16Vector("f16_vector", vector);
    EXPECT_TRUE(status.IsOk());

    // keep even primary keys in the first half of the pk range. Pages beyond the
    // first half are entirely filtered out (exercising the fully-filtered continue),
    // while pages in the first half are partially filtered (even rows kept).
    arguments.SetExternalFilterFunc([row_count](milvus::SingleResult& page) {
        std::vector<uint64_t> keep_indices;
        auto ids = page.Ids();
        EXPECT_TRUE(ids.IsIntegerID());
        for (uint64_t i = 0; i < page.GetRowCount(); i++) {
            auto pk = ids.IntIDArray().at(i);
            if (pk % 2 == 0 && pk < row_count / 2) {
                keep_indices.push_back(i);
            }
        }
        return page.FilterRows(keep_indices);
    });

    milvus::SearchIteratorPtr iterator;
    status = client->SearchIterator(arguments, iterator);
    EXPECT_TRUE(status.IsOk());

    uint64_t returned_count = 0;
    while (true) {
        milvus::SingleResult batch_results;
        status = iterator->Next(batch_results);
        EXPECT_TRUE(status.IsOk());
        auto batch_count = batch_results.GetRowCount();
        if (batch_count == 0) {
            break;
        }
        returned_count += batch_count;

        auto ids = batch_results.Ids();
        for (uint64_t i = 0; i < static_cast<uint64_t>(batch_count); i++) {
            auto pk = ids.IntIDArray().at(i);
            EXPECT_EQ(pk % 2, 0) << "odd row leaked through the external filter";
            EXPECT_LT(pk, row_count / 2) << "row beyond the kept range leaked through the external filter";
        }
    }
    EXPECT_EQ(returned_count, static_cast<uint64_t>(row_count) / 4);
}

TEST_F(MilvusMockedTest, SearchIteratorV2AppliesExternalFilterPerPage) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    auto status = client_->Connect(connect_param);
    EXPECT_TRUE(status.IsOk());

    DoSearchIteratorWithExternalFilter(service_, client_, false);
}

TEST_F(MilvusMockedTest, SearchIteratorV1AppliesExternalFilterPerPage) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    auto status = client_->Connect(connect_param);
    EXPECT_TRUE(status.IsOk());

    DoSearchIteratorWithExternalFilter(service_, client_, true);
}

namespace {
const std::string kCursorCollection = "CursorCollection";
const std::string kCursorToken = "4ea6247d-4b47-4e95-a65c-3bca62bbf7c1";

std::string
CursorParam(const SearchRequest& request, const std::string& key) {
    for (const auto& pair : request.search_params()) {
        if (pair.key() == key) {
            return pair.value();
        }
    }
    return "";
}

void
ExpectCursorSchema(testing::StrictMock<milvus::MilvusMockedService>& service,
                   milvus::DataType pk_type = milvus::DataType::INT64) {
    EXPECT_CALL(service, DescribeCollection(_, _, _))
        .WillOnce(
            [pk_type](::grpc::ServerContext*, const DescribeCollectionRequest*, DescribeCollectionResponse* response) {
                milvus::CollectionSchema schema(kCursorCollection);
                schema.AddField(milvus::FieldSchema("id", pk_type, "", true, false));
                schema.AddField(milvus::FieldSchema("vector", milvus::DataType::FLOAT_VECTOR).WithDimension(2));
                response->set_collectionid(100);
                milvus::ConvertCollectionSchema(schema, *response->mutable_schema());
                return ::grpc::Status{};
            });
}

milvus::SearchIteratorArguments
CursorArgs(uint64_t batch, int64_t limit, bool opt_in) {
    milvus::SearchIteratorArguments args;
    args.SetCollectionName(kCursorCollection);
    args.SetMetricType(milvus::MetricType::COSINE);
    args.SetBatchSize(batch);
    args.SetLimit(limit);
    args.AddFloatVector("vector", {0.1f, 0.2f});
    if (opt_in) {
        args.AddExtraParam("search_iter_cursor_version", "2");
    }
    return args;
}

void
FillCursorReply(SearchResults* response, const std::vector<int64_t>& ids, const std::vector<float>& scores, uint64_t ts,
                const std::string& version = "", bool supports_v2 = true) {
    response->set_session_ts(ts);
    auto* data = response->mutable_results();
    data->set_num_queries(1);
    data->set_top_k(static_cast<int64_t>(ids.size()));
    data->add_topks(static_cast<int64_t>(ids.size()));
    data->set_primary_field_name("id");
    for (auto id : ids) {
        data->mutable_ids()->mutable_int_id()->add_data(id);
    }
    for (auto score : scores) {
        data->add_scores(score);
    }
    if (supports_v2) {
        auto* info = data->mutable_search_iterator_v2_results();
        info->set_token(kCursorToken);
        info->set_last_bound(scores.empty() ? 0.0f : scores.back());
    }
    if (!version.empty()) {
        auto* extra = response->mutable_status()->mutable_extra_info();
        (*extra)["search_iter_cursor_version"] = version;
        if (!ids.empty()) {
            (*extra)["search_iter_last_pk_type"] = "int64";
            (*extra)["search_iter_last_pk"] = std::to_string(ids.back());
        }
    }
}
}  // namespace

TEST_F(MilvusMockedTest, SearchIteratorDefaultUsesRealBatchWithoutPkOptIn) {
    ASSERT_TRUE(client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()}).IsOk());
    ExpectCursorSchema(service_);
    EXPECT_CALL(service_, Search(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            EXPECT_EQ(CursorParam(*request, milvus::TOPK), "2");
            EXPECT_TRUE(CursorParam(*request, "search_iter_cursor_version").empty());
            FillCursorReply(response, {1, 2}, {0.9f, 0.8f}, 301);
            return ::grpc::Status{};
        });
    auto args = CursorArgs(2, 2, false);
    milvus::SearchIteratorPtr iterator;
    ASSERT_TRUE(client_->SearchIterator(args, iterator).IsOk());
    EXPECT_EQ(args.CollectionID(), 0);
    milvus::SingleResult page;
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.GetRowCount(), 2);
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.GetRowCount(), 0);
}

TEST_F(MilvusMockedTest, SearchIteratorPkCursorCachesFirstPageAndPinsSnapshot) {
    ASSERT_TRUE(client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()}).IsOk());
    ExpectCursorSchema(service_);
    EXPECT_CALL(service_, Search(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            EXPECT_EQ(CursorParam(*request, "search_iter_cursor_version"), "2");
            EXPECT_EQ(CursorParam(*request, milvus::TOPK), "2");
            auto nested = nlohmann::json::parse(CursorParam(*request, milvus::PARAMS));
            EXPECT_FALSE(nested.contains("search_iter_cursor_version"));
            EXPECT_EQ(request->guarantee_timestamp(), 0);
            FillCursorReply(response, {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()},
                            {0.9f, 0.8f}, 301, "2");
            return ::grpc::Status{};
        })
        .WillOnce([](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            EXPECT_EQ(request->guarantee_timestamp(), 301);
            EXPECT_EQ(request->db_name(), "default");
            EXPECT_EQ(CursorParam(*request, "search_iter_last_pk_type"), "int64");
            EXPECT_EQ(CursorParam(*request, "search_iter_last_pk"), "9223372036854775807");
            EXPECT_EQ(CursorParam(*request, milvus::ITER_SEARCH_ID_KEY), kCursorToken);
            FillCursorReply(response, {3, 4}, {0.7f, 0.6f}, 0, "2");
            return ::grpc::Status{};
        });
    auto args = CursorArgs(2, 3, true);
    milvus::SearchIteratorPtr iterator;
    ASSERT_TRUE(client_->SearchIterator(args, iterator).IsOk());
    args.SetBatchSize(99);
    args.SetFilter("id < 0");
    EXPECT_CALL(service_, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::ConnectRequest*,
                     milvus::proto::milvus::ConnectResponse*) { return ::grpc::Status{}; });
    ASSERT_TRUE(client_->UseDatabase("other").IsOk());
    milvus::SingleResult first, last;
    ASSERT_TRUE(iterator->Next(first).IsOk());
    EXPECT_EQ(first.Ids().IntIDArray(),
              (std::vector<int64_t>{std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()}));
    ASSERT_TRUE(iterator->Next(last).IsOk());
    EXPECT_EQ(last.GetRowCount(), 1);
    ASSERT_TRUE(iterator->Next(last).IsOk());
    EXPECT_EQ(last.GetRowCount(), 0);
}

TEST_F(MilvusMockedTest, SearchIteratorPkCursorRetainsRawPageOnFilterErrorOrThrow) {
    ASSERT_TRUE(client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()}).IsOk());
    ExpectCursorSchema(service_);
    EXPECT_CALL(service_, Search(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const SearchRequest*, SearchResults* response) {
            FillCursorReply(response, {1}, {0.9f}, 301, "2");
            return ::grpc::Status{};
        });
    auto args = CursorArgs(1, 1, true);
    int calls = 0;
    args.SetExternalFilterFunc([&calls](milvus::SingleResult& result) {
        ++calls;
        if (calls == 1) {
            result.Clear();
            return milvus::Status{milvus::StatusCode::UNKNOWN_ERROR, "filter failed"};
        }
        if (calls == 2) {
            result.Clear();
            throw std::runtime_error("filter threw");
        }
        return milvus::Status::OK();
    });
    milvus::SearchIteratorPtr iterator;
    ASSERT_TRUE(client_->SearchIterator(args, iterator).IsOk());
    milvus::SingleResult page;
    EXPECT_FALSE(iterator->Next(page).IsOk());
    EXPECT_FALSE(iterator->Next(page).IsOk());
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.GetRowCount(), 1);
    EXPECT_EQ(page.Ids().IntIDArray(), (std::vector<int64_t>{1}));
}

TEST_F(MilvusMockedTest, SearchIteratorPkCursorDuplicatesDoNotConsumeLimitAndFilterRetryRetainsPage) {
    ASSERT_TRUE(client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()}).IsOk());
    ExpectCursorSchema(service_);
    EXPECT_CALL(service_, Search(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const SearchRequest*, SearchResults* response) {
            FillCursorReply(response, {1}, {0.9f}, 301, "2");
            return ::grpc::Status{};
        })
        .WillOnce([](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            EXPECT_EQ(CursorParam(*request, "search_iter_last_pk"), "1");
            EXPECT_FLOAT_EQ(std::stof(CursorParam(*request, milvus::ITER_SEARCH_LAST_BOUND_KEY)), 0.9f);
            FillCursorReply(response, {1}, {0.8f}, 0, "2");
            return ::grpc::Status{};
        })
        .WillOnce([](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            EXPECT_EQ(CursorParam(*request, "search_iter_last_pk"), "1");
            EXPECT_FLOAT_EQ(std::stof(CursorParam(*request, milvus::ITER_SEARCH_LAST_BOUND_KEY)), 0.8f);
            FillCursorReply(response, {2}, {0.7f}, 0, "2");
            return ::grpc::Status{};
        });
    auto args = CursorArgs(1, 2, true);
    bool failed = false;
    args.SetExternalFilterFunc([&failed](milvus::SingleResult& page) {
        if (page.Ids().IntIDArray().front() == 2 && !failed) {
            failed = true;
            page.Clear();
            return milvus::Status{milvus::StatusCode::UNKNOWN_ERROR, "retry this raw page"};
        }
        return milvus::Status::OK();
    });
    milvus::SearchIteratorPtr iterator;
    ASSERT_TRUE(client_->SearchIterator(args, iterator).IsOk());
    milvus::SingleResult page;
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.Ids().IntIDArray(), (std::vector<int64_t>{1}));
    EXPECT_FALSE(iterator->Next(page).IsOk());
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.Ids().IntIDArray(), (std::vector<int64_t>{2}));
    EXPECT_EQ(page.Scores(), (std::vector<float>{0.7f}));
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.GetRowCount(), 0);
}

TEST_F(MilvusMockedTest, SearchIteratorDistanceModePreservesRepeatedPrimaryKeys) {
    ASSERT_TRUE(client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()}).IsOk());
    ExpectCursorSchema(service_);
    EXPECT_CALL(service_, Search(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const SearchRequest*, SearchResults* response) {
            FillCursorReply(response, {1}, {0.9f}, 301);
            return ::grpc::Status{};
        })
        .WillOnce([](::grpc::ServerContext*, const SearchRequest*, SearchResults* response) {
            FillCursorReply(response, {1}, {0.8f}, 0);
            return ::grpc::Status{};
        });
    auto args = CursorArgs(1, 2, false);
    milvus::SearchIteratorPtr iterator;
    ASSERT_TRUE(client_->SearchIterator(args, iterator).IsOk());
    milvus::SingleResult page;
    ASSERT_TRUE(iterator->Next(page).IsOk());
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.Ids().IntIDArray(), (std::vector<int64_t>{1}));
    EXPECT_EQ(page.Scores(), (std::vector<float>{0.8f}));
}

TEST_F(MilvusMockedTest, SearchIteratorPkCursorDeduplicatesExactVarcharKeys) {
    ASSERT_TRUE(client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()}).IsOk());
    ExpectCursorSchema(service_, milvus::DataType::VARCHAR);
    const std::string quoted = "quoted\"\\中文";
    const auto fill = [&quoted](SearchResults* response, bool first) {
        FillCursorReply(response, {1, 2}, first ? std::vector<float>{0.9f, 0.8f} : std::vector<float>{0.7f, 0.6f},
                        first ? 301 : 0, "2");
        auto* ids = response->mutable_results()->mutable_ids()->mutable_str_id();
        ids->add_data(first ? "" : quoted);
        ids->add_data(first ? quoted : "last");
        auto* extra = response->mutable_status()->mutable_extra_info();
        (*extra)["search_iter_last_pk_type"] = "varchar";
        (*extra)["search_iter_last_pk"] = ids->data(1);
    };
    EXPECT_CALL(service_, Search(_, _, _))
        .WillOnce([&fill](::grpc::ServerContext*, const SearchRequest*, SearchResults* response) {
            fill(response, true);
            return ::grpc::Status{};
        })
        .WillOnce([&fill, &quoted](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            EXPECT_EQ(CursorParam(*request, "search_iter_last_pk"), quoted);
            fill(response, false);
            return ::grpc::Status{};
        });
    auto args = CursorArgs(2, 3, true);
    milvus::SearchIteratorPtr iterator;
    ASSERT_TRUE(client_->SearchIterator(args, iterator).IsOk());
    milvus::SingleResult page;
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.Ids().StrIDArray(), (std::vector<std::string>{"", quoted}));
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.Ids().StrIDArray(), (std::vector<std::string>{"last"}));
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.GetRowCount(), 0);
}

TEST_F(MilvusMockedTest, SearchIteratorPkCursorRejectsChangedModeAndBadRawShapeTransactionally) {
    ASSERT_TRUE(client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()}).IsOk());
    ExpectCursorSchema(service_);
    std::string retry_request;
    int calls = 0;
    EXPECT_CALL(service_, Search(_, _, _))
        .Times(6)
        .WillRepeatedly([&](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            ++calls;
            if (calls == 1) {
                FillCursorReply(response, {1}, {0.9f}, 301, "2");
                return ::grpc::Status{};
            }
            if (calls == 2) {
                retry_request = request->SerializeAsString();
            }
            EXPECT_EQ(request->SerializeAsString(), retry_request);
            EXPECT_EQ(request->guarantee_timestamp(), 301);
            FillCursorReply(response, {2}, {0.8f}, 0, calls == 2 ? "" : "2");
            if (calls == 3) {
                (*response->mutable_status()->mutable_extra_info())["search_iter_last_pk"] = "3";
            }
            if (calls == 4) {
                response->mutable_results()->mutable_search_iterator_v2_results()->set_last_bound(0.7f);
            }
            if (calls == 5) {
                response->mutable_results()->clear_scores();
            }
            return ::grpc::Status{};
        });
    auto args = CursorArgs(1, 2, true);
    milvus::SearchIteratorPtr iterator;
    ASSERT_TRUE(client_->SearchIterator(args, iterator).IsOk());
    milvus::SingleResult page;
    ASSERT_TRUE(iterator->Next(page).IsOk());
    for (int i = 0; i < 4; ++i) {
        EXPECT_FALSE(iterator->Next(page).IsOk());
    }
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.Ids().IntIDArray(), (std::vector<int64_t>{2}));
}

TEST_F(MilvusMockedTest, SearchIteratorPkCursorSupportsEmptyAndQuotedVarcharKeys) {
    ASSERT_TRUE(client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()}).IsOk());
    ExpectCursorSchema(service_, milvus::DataType::VARCHAR);
    const std::string quoted = "a\"b\\c\n中文";
    EXPECT_CALL(service_, Search(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const SearchRequest*, SearchResults* response) {
            FillCursorReply(response, {1}, {0.9f}, 301, "2");
            response->mutable_results()->mutable_ids()->mutable_str_id()->add_data("");
            (*response->mutable_status()->mutable_extra_info())["search_iter_last_pk_type"] = "varchar";
            (*response->mutable_status()->mutable_extra_info())["search_iter_last_pk"] = "";
            return ::grpc::Status{};
        })
        .WillOnce([quoted](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            EXPECT_EQ(CursorParam(*request, "search_iter_last_pk_type"), "varchar");
            EXPECT_EQ(CursorParam(*request, "search_iter_last_pk"), "");
            FillCursorReply(response, {1}, {0.8f}, 0, "2");
            response->mutable_results()->mutable_ids()->mutable_str_id()->add_data(quoted);
            (*response->mutable_status()->mutable_extra_info())["search_iter_last_pk_type"] = "varchar";
            (*response->mutable_status()->mutable_extra_info())["search_iter_last_pk"] = quoted;
            return ::grpc::Status{};
        })
        .WillOnce([quoted](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            EXPECT_EQ(CursorParam(*request, "search_iter_last_pk"), quoted);
            FillCursorReply(response, {}, {}, 0, "2");
            return ::grpc::Status{};
        });
    auto args = CursorArgs(1, 3, true);
    milvus::SearchIteratorPtr iterator;
    ASSERT_TRUE(client_->SearchIterator(args, iterator).IsOk());
    milvus::SingleResult page;
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.Ids().StrIDArray(), (std::vector<std::string>{""}));
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.Ids().StrIDArray(), (std::vector<std::string>{quoted}));
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.GetRowCount(), 0);
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.GetRowCount(), 0);
}

TEST_F(MilvusMockedTest, SearchIteratorLegacyFallbackReusesEquivalentRealFirstPage) {
    ASSERT_TRUE(client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()}).IsOk());
    ExpectCursorSchema(service_);
    auto args = CursorArgs(2, 2, false);
    args.SetFilter("id >= 0");
    args.AddExtraParam(milvus::RADIUS, "0.5");
    args.AddExtraParam(milvus::RANGE_FILTER, "1.0");
    EXPECT_CALL(service_, Search(_, _, _))
        .WillOnce([&](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            SearchRequest expected;
            auto equivalent = args;
            equivalent.SetLimit(2);
            EXPECT_TRUE(milvus::ConvertSearchRequest(equivalent, "default", expected, "", "127.0.0.1").IsOk());
            EXPECT_EQ(request->dsl(), expected.dsl());
            EXPECT_EQ(request->placeholder_group(), expected.placeholder_group());
            for (const auto& key : {milvus::TOPK, milvus::METRIC_TYPE, milvus::RADIUS, milvus::RANGE_FILTER}) {
                EXPECT_EQ(CursorParam(*request, key), CursorParam(expected, key));
            }
            FillCursorReply(response, {1, 2}, {0.9f, 0.8f}, 301, "", false);
            return ::grpc::Status{};
        });
    milvus::SearchIteratorPtr iterator;
    ASSERT_TRUE(client_->SearchIterator(args, iterator).IsOk());
    milvus::SingleResult page;
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.GetRowCount(), 2);
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.GetRowCount(), 0);
}

TEST_F(MilvusMockedTest, SearchIteratorLegacyFallbackEmptyFirstPageFinishes) {
    ASSERT_TRUE(client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()}).IsOk());
    ExpectCursorSchema(service_);
    EXPECT_CALL(service_, Search(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const SearchRequest*, SearchResults* response) {
            FillCursorReply(response, {}, {}, 301, "", false);
            return ::grpc::Status{};
        });
    auto args = CursorArgs(2, 10, false);
    milvus::SearchIteratorPtr iterator;
    ASSERT_TRUE(client_->SearchIterator(args, iterator).IsOk());
    milvus::SingleResult page;
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.GetRowCount(), 0);
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.GetRowCount(), 0);
}

TEST_F(MilvusMockedTest, SearchIteratorManualDistanceContinuationPreservesLegacyMode) {
    ASSERT_TRUE(client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()}).IsOk());
    ExpectCursorSchema(service_);
    EXPECT_CALL(service_, Search(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            EXPECT_EQ(CursorParam(*request, milvus::ITER_SEARCH_ID_KEY), kCursorToken);
            EXPECT_EQ(CursorParam(*request, milvus::ITER_SEARCH_LAST_BOUND_KEY), "0.7");
            EXPECT_TRUE(CursorParam(*request, "search_iter_cursor_version").empty());
            FillCursorReply(response, {2}, {0.6f}, 301);
            return ::grpc::Status{};
        });
    auto args = CursorArgs(1, 1, false);
    args.AddExtraParam(milvus::ITER_SEARCH_ID_KEY, kCursorToken);
    args.AddExtraParam(milvus::ITER_SEARCH_LAST_BOUND_KEY, "0.7");
    milvus::SearchIteratorPtr iterator;
    ASSERT_TRUE(client_->SearchIterator(args, iterator).IsOk());
    milvus::SingleResult page;
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.GetRowCount(), 1);
}

TEST_F(MilvusMockedTest, SearchIteratorPkCursorRejectsPartialLegacyResumeBeforeSearch) {
    ASSERT_TRUE(client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()}).IsOk());
    ExpectCursorSchema(service_);
    auto args = CursorArgs(1, 1, true);
    args.AddExtraParam(milvus::ITER_SEARCH_ID_KEY, kCursorToken);
    args.AddExtraParam(milvus::ITER_SEARCH_LAST_BOUND_KEY, "0.7");
    milvus::SearchIteratorPtr iterator;
    EXPECT_EQ(client_->SearchIterator(args, iterator).Code(), milvus::StatusCode::INVALID_ARGUMENT);
}

TEST_F(MilvusMockedTest, SearchIteratorRejectsUnknownRequestedCursorVersionBeforeSearch) {
    ASSERT_TRUE(client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()}).IsOk());
    ExpectCursorSchema(service_);
    auto args = CursorArgs(1, 1, false);
    args.AddExtraParam("search_iter_cursor_version", "3");
    milvus::SearchIteratorPtr iterator;
    EXPECT_EQ(client_->SearchIterator(args, iterator).Code(), milvus::StatusCode::INVALID_ARGUMENT);
}

TEST_F(MilvusMockedTest, SearchIteratorRejectsUnrequestedPkResponseWithoutLegacyFallback) {
    ASSERT_TRUE(client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()}).IsOk());
    ExpectCursorSchema(service_);
    EXPECT_CALL(service_, Search(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const SearchRequest*, SearchResults* response) {
            FillCursorReply(response, {1}, {0.9f}, 301, "2");
            return ::grpc::Status{};
        });
    auto args = CursorArgs(1, 1, false);
    milvus::SearchIteratorPtr iterator;
    EXPECT_EQ(client_->SearchIterator(args, iterator).Code(), milvus::StatusCode::UNKNOWN_ERROR);
}

TEST_F(MilvusMockedTest, SearchIteratorPkResponseNeedsInitialSnapshotAndToken) {
    ASSERT_TRUE(client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()}).IsOk());
    ExpectCursorSchema(service_);
    EXPECT_CALL(service_, Search(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const SearchRequest*, SearchResults* response) {
            FillCursorReply(response, {1}, {0.9f}, 0, "2");
            return ::grpc::Status{};
        });
    auto args = CursorArgs(1, 1, true);
    milvus::SearchIteratorPtr iterator;
    EXPECT_EQ(client_->SearchIterator(args, iterator).Code(), milvus::StatusCode::UNKNOWN_ERROR);
}

TEST_F(MilvusMockedTest, SearchIteratorPkShortPagesContinueAndSurplusCacheAvoidsRpc) {
    ASSERT_TRUE(client_->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()}).IsOk());
    ExpectCursorSchema(service_);
    int calls = 0;
    EXPECT_CALL(service_, Search(_, _, _))
        .Times(3)
        .WillRepeatedly([&](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            ++calls;
            EXPECT_EQ(CursorParam(*request, milvus::TOPK), "2");
            FillCursorReply(response, {calls}, {1.0f - 0.1f * static_cast<float>(calls)}, calls == 1 ? 301 : 0, "2");
            return ::grpc::Status{};
        });
    auto args = CursorArgs(2, 3, true);
    milvus::SearchIteratorPtr iterator;
    ASSERT_TRUE(client_->SearchIterator(args, iterator).IsOk());
    milvus::SingleResult page;
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.GetRowCount(), 2);
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.GetRowCount(), 1);
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.GetRowCount(), 0);
}

TEST_F(UnconnectMilvusMockedTest, SearchIteratorV2FacadeSupportsExplicitPkCursorWithoutMutatingRequest) {
    EXPECT_CALL(service_, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::ConnectRequest*,
                     milvus::proto::milvus::ConnectResponse*) { return ::grpc::Status{}; });
    auto client = milvus::MilvusClientV2::Create();
    ASSERT_TRUE(client->Connect(milvus::ConnectParam{"127.0.0.1", server_.ListenPort()}).IsOk());
    ExpectCursorSchema(service_);
    EXPECT_CALL(service_, Search(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            EXPECT_EQ(CursorParam(*request, "search_iter_cursor_version"), "2");
            FillCursorReply(response, {1}, {0.9f}, 301, "2");
            return ::grpc::Status{};
        });
    milvus::SearchIteratorRequest request;
    request.SetCollectionName(kCursorCollection);
    request.SetAnnsField("vector");
    request.SetMetricType(milvus::MetricType::COSINE);
    request.SetBatchSize(1);
    request.SetLimit(1);
    request.AddFloatVector({0.1f, 0.2f});
    request.AddExtraParam("search_iter_cursor_version", "2");
    milvus::SearchIteratorPtr iterator;
    ASSERT_TRUE(client->SearchIterator(request, iterator).IsOk());
    EXPECT_EQ(request.CollectionID(), 0);
    EXPECT_TRUE(request.DatabaseName().empty());
    milvus::SingleResult page;
    ASSERT_TRUE(iterator->Next(page).IsOk());
    EXPECT_EQ(page.GetRowCount(), 1);
}
