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

using ::milvus::proto::milvus::DescribeCollectionRequest;
using ::milvus::proto::milvus::DescribeCollectionResponse;
using ::milvus::proto::milvus::SearchRequest;
using ::milvus::proto::milvus::SearchResults;

using ::testing::_;
using ::testing::Property;
using ::testing::UnorderedElementsAreArray;

milvus::Status
DoSearchIterator(testing::StrictMock<milvus::MilvusMockedService>& service, milvus::MilvusClientPtr& client, bool v1) {
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
    const milvus::ConsistencyLevel level = milvus::ConsistencyLevel::STRONG;
    const uint64_t batch_size = 3000;
    const int64_t limit = row_count;
    uint64_t current_poz = 0;
    bool probe_compability = true;
    EXPECT_CALL(service, Search(_, _, _))
        .WillRepeatedly([&](::grpc::ServerContext*, const SearchRequest* request, SearchResults* response) {
            auto token = v1 ? "" : "dummy";
            response->mutable_results()->mutable_search_iterator_v2_results()->set_token(token);
            if (probe_compability) {
                probe_compability = false;
                return ::grpc::Status{};
            }

            auto params = request->search_params();
            for (auto pair : params) {
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

    DoSearchIterator(service_, client_, true);
}

TEST_F(MilvusMockedTest, SearchIteratorV2) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    auto status = client_->Connect(connect_param);
    EXPECT_TRUE(status.IsOk());

    DoSearchIterator(service_, client_, false);
}