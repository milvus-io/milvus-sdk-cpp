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

#include <memory>

#include "../mocks/MilvusMockedTest.h"
#include "../mocks/Utils.h"
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
    bool probe_compability = true;
    constexpr uint64_t probe_session_ts = 123456;
    EXPECT_CALL(service, Search(_, _, _))
        .WillRepeatedly([&](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            auto token = v1 ? "" : "dummy";
            response->mutable_results()->mutable_search_iterator_v2_results()->set_token(token);
            if (probe_compability) {
                probe_compability = false;
                if (!v1) {
                    response->set_session_ts(probe_session_ts);
                }
                return ::grpc::Status{};
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
                EXPECT_EQ(request->guarantee_timestamp(), probe_session_ts);
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

TEST_F(MilvusMockedTest, SearchIteratorV2PinsProbeTimestampForSessionConsistency) {
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
            response->mutable_results()->mutable_search_iterator_v2_results()->set_token("dummy");
            return ::grpc::Status{};
        })
        .WillOnce([iterator_session_ts](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            EXPECT_EQ(request->guarantee_timestamp(), iterator_session_ts);
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
    bool probe_compability = true;
    EXPECT_CALL(service, Search(_, _, _))
        .WillRepeatedly([&](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            auto token = use_v1 ? "" : "dummy";
            response->mutable_results()->mutable_search_iterator_v2_results()->set_token(token);
            if (probe_compability) {
                probe_compability = false;
                if (!use_v1) {
                    response->set_session_ts(123456);
                }
                return ::grpc::Status{};
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
