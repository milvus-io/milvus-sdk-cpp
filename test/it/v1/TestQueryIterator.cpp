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
using ::milvus::proto::milvus::QueryRequest;
using ::milvus::proto::milvus::QueryResults;

using ::testing::_;
using ::testing::Property;
using ::testing::UnorderedElementsAreArray;

TEST_F(MilvusMockedTest, QueryIterator) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

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

    EXPECT_CALL(service_,
                DescribeCollection(_, Property(&DescribeCollectionRequest::collection_name, collection_name), _))
        .WillOnce([&](::grpc::ServerContext*, const DescribeCollectionRequest*, DescribeCollectionResponse* response) {
            response->set_collectionid(100);
            response->set_shards_num(2);
            response->set_created_timestamp(1111);
            auto proto_schema = response->mutable_schema();
            milvus::ConvertCollectionSchema(collection_schema, *proto_schema);
            return ::grpc::Status{};
        });

    const uint64_t batch_size = 300;
    const int64_t limit = row_count;
    const int64_t offset = 17000;
    uint64_t current_poz = 0;
    EXPECT_CALL(service_, Query(_, _, _))
        .WillRepeatedly([&](::grpc::ServerContext*, const QueryRequest* request, QueryResults* response) {
            auto params = request->query_params();
            uint64_t session_ts = 999999;
            bool is_seek = false;
            uint64_t query_limit = 0;
            for (const auto& pair : params) {
                if (pair.key() == milvus::LIMIT) {
                    query_limit = std::atoi(pair.value().c_str());
                    if (query_limit == 1) {
                        // query iterator init will call query to get a timestamp
                        response->set_session_ts(session_ts);
                        return ::grpc::Status{};
                    }
                }
                if (pair.key() == milvus::ITERATOR_FIELD && pair.value() == "False") {
                    is_seek = true;
                }
            }
            EXPECT_EQ(request->collection_name(), collection_name);
            EXPECT_EQ(request->guarantee_timestamp(), session_ts);
            EXPECT_EQ(request->consistency_level(), milvus::proto::common::ConsistencyLevel::Bounded);

            if (is_seek) {
                EXPECT_EQ(request->output_fields_size(), 0);
                milvus::FieldDataPtr id_field = fields_data.at(0);
                milvus::FieldDataPtr offset_id;
                auto status = milvus::CopyFieldData(id_field, current_poz, current_poz + query_limit, offset_id);
                EXPECT_TRUE(status.IsOk());
                milvus::FieldDataSchema bridge(offset_id, nullptr);
                milvus::proto::schema::FieldData data;
                status = milvus::CreateProtoFieldData(bridge, data);
                EXPECT_TRUE(status.IsOk());
                auto* mutable_fields = response->mutable_fields_data();
                mutable_fields->Add(std::move(data));
                current_poz += offset_id->Count();
                return ::grpc::Status{};
            }

            EXPECT_THAT(request->output_fields(), UnorderedElementsAreArray(field_names));
            EXPECT_EQ(query_limit, batch_size);

            auto from = current_poz;
            auto to = from + batch_size;
            if (current_poz > offset + 2 * batch_size) {
                // let the iterator run into the cache logic
                to = from + 2 * batch_size + 5;
                current_poz += 2 * batch_size;
            } else {
                current_poz += batch_size;
            }
            if (from >= static_cast<uint64_t>(row_count)) {
                return ::grpc::Status{};
            }

            auto* mutable_fields = response->mutable_fields_data();
            for (const auto& field_data : fields_data) {
                milvus::FieldDataPtr page_data;
                auto status = milvus::CopyFieldData(field_data, from, to, page_data);
                EXPECT_TRUE(status.IsOk());
                milvus::FieldDataSchema bridge(page_data, nullptr);
                milvus::proto::schema::FieldData data;
                status = milvus::CreateProtoFieldData(bridge, data);
                EXPECT_TRUE(status.IsOk());
                mutable_fields->Add(std::move(data));
            }
            return ::grpc::Status{};
        });

    milvus::QueryIteratorArguments arguments{};
    arguments.SetBatchSize(batch_size);
    arguments.SetOffset(offset);
    arguments.SetLimit(limit);
    arguments.SetCollectionName(collection_name);
    arguments.SetFilter("id >= 0");
    arguments.SetConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);
    for (const auto& name : field_names) {
        arguments.AddOutputField(name);
    }

    milvus::QueryIteratorPtr iterator;
    auto status = client_->QueryIterator(arguments, iterator);
    EXPECT_TRUE(status.IsOk());

    milvus::EntityRows total_rows;
    while (true) {
        milvus::QueryResults batch_results;
        status = iterator->Next(batch_results);
        EXPECT_TRUE(status.IsOk());
        if (batch_results.GetRowCount() == 0) {
            // std::cout << "query iteration finished" << std::endl;
            break;
        }
        // std::cout << std::to_string(batch_results.GetRowCount()) + " rows fetched" << std::endl;

        milvus::EntityRows batch_rows;
        status = batch_results.OutputRows(batch_rows);
        EXPECT_TRUE(status.IsOk());
        std::copy(batch_rows.begin(), batch_rows.end(), std::back_inserter(total_rows));
    }
    EXPECT_EQ(total_rows.size(), row_count - offset);

    std::vector<milvus::FieldDataPtr> expected_fields;
    status = milvus::CopyFieldsData(fields_data, offset, offset + limit, expected_fields);
    EXPECT_TRUE(status.IsOk());

    milvus::QueryResults expected_results{std::move(expected_fields), arguments.OutputFields()};
    milvus::EntityRows expected_rows;
    status = expected_results.OutputRows(expected_rows);
    EXPECT_TRUE(status.IsOk());

    EXPECT_EQ(total_rows.size(), expected_rows.size());
    for (auto i = 0; i < total_rows.size(); i++) {
        EXPECT_EQ(total_rows.at(i), expected_rows.at(i));
        if (total_rows.at(i) != expected_rows.at(i)) {
            break;
        }
    }
}
