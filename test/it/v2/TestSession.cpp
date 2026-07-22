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
#include <string>

#include "../mocks/MilvusMockedTest.h"
#include "../mocks/Utils.h"
#include "milvus/MilvusClientV2.h"
#include "utils/TypeUtils.h"

using ::testing::_;
using ::testing::Property;

namespace {

milvus::MilvusClientV2Ptr
CreateConnectedClient(testing::StrictMock<milvus::MilvusMockedService>& service, uint16_t port) {
    EXPECT_CALL(service, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::ConnectRequest*,
                     milvus::proto::milvus::ConnectResponse*) { return ::grpc::Status{}; });

    auto client = milvus::MilvusClientV2::Create();
    auto status = client->Connect(milvus::ConnectParam{"127.0.0.1", port});
    EXPECT_TRUE(status.IsOk());
    return client;
}

int
CountParam(const google::protobuf::RepeatedPtrField<milvus::proto::common::KeyValuePair>& params,
           const std::string& key, const std::string& value) {
    int count = 0;
    for (const auto& param : params) {
        if (param.key() == key) {
            ++count;
            EXPECT_EQ(param.value(), value);
        }
    }
    return count;
}

void
FillSearchResults(milvus::proto::milvus::SearchResults* response) {
    response->mutable_status()->set_code(milvus::proto::common::ErrorCode::Success);
    auto* results = response->mutable_results();
    results->set_num_queries(1);
    results->set_primary_field_name("id");
    results->mutable_topks()->Add(0);
}

}  // namespace

TEST_F(UnconnectMilvusMockedTest, V2SessionValidationAndClose) {
    auto client = CreateConnectedClient(service_, server_.ListenPort());

    milvus::MilvusClientV2SessionPtr session;
    ASSERT_TRUE(client->Session("existing", session).IsOk());
    auto status = client->Session("", session);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(session, nullptr);

    status = client->Session("cluster-a", session);
    ASSERT_TRUE(status.IsOk());
    ASSERT_NE(session, nullptr);
    EXPECT_EQ(session->ClusterID(), "cluster-a");

    std::weak_ptr<milvus::MilvusClientV2> weak_client = client;
    session->Close();
    session->Close();
    client.reset();
    EXPECT_TRUE(weak_client.expired());

    milvus::QueryRequest request;
    milvus::QueryResponse response;
    status = session->Query(request, response);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::NOT_CONNECTED);
    EXPECT_EQ(status.Message(), "MilvusClient session is closed");
}

TEST_F(UnconnectMilvusMockedTest, V2IteratorsRejectOrderByBeforeRpc) {
    auto client = CreateConnectedClient(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeCollection(_, _, _)).Times(0);
    EXPECT_CALL(service_, Query(_, _, _)).Times(0);
    EXPECT_CALL(service_, Search(_, _, _)).Times(0);

    milvus::QueryIteratorRequest query_request;
    query_request.WithCollectionName("foo").AddOrderByField(milvus::OrderByField("price"));
    milvus::QueryIteratorPtr query_iterator;
    auto status = client->QueryIterator(query_request, query_iterator);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.Message(), "ORDER BY with iterator is not supported");

    milvus::SearchIteratorRequest search_request;
    search_request.WithCollectionName("foo").AddOrderByField(milvus::OrderByField("price"));
    milvus::SearchIteratorPtr search_iterator;
    status = client->SearchIterator(search_request, search_iterator);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.Message(), "ORDER BY with iterator is not supported");
}

TEST_F(UnconnectMilvusMockedTest, V2SearchIteratorRejectsIDs) {
    auto client = CreateConnectedClient(service_, server_.ListenPort());

    milvus::SearchIteratorRequest request;
    static_cast<milvus::SearchRequest&>(request).WithIDs(std::vector<int64_t>{1});
    milvus::SearchIteratorPtr iterator;

    auto status = client->SearchIterator(request, iterator);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(status.Message(), "Search iterator does not support IDs as search targets");
    EXPECT_EQ(iterator, nullptr);
}

TEST_F(UnconnectMilvusMockedTest, V2SessionUnaryRoutingAndIsolation) {
    auto client = CreateConnectedClient(service_, server_.ListenPort());
    milvus::MilvusClientV2SessionPtr session_a;
    milvus::MilvusClientV2SessionPtr session_b;
    ASSERT_TRUE(client->Session("cluster-a", session_a).IsOk());
    ASSERT_TRUE(client->Session("cluster-b", session_b).IsOk());

    EXPECT_CALL(service_, Search(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::SearchRequest* request,
                     milvus::proto::milvus::SearchResults* response) {
            EXPECT_EQ(CountParam(request->search_params(), "cluster_id", "cluster-a"), 1);
            FillSearchResults(response);
            return ::grpc::Status{};
        });
    milvus::SearchRequest search_request;
    search_request.WithCollectionName("foo").WithLimit(1).AddFloatVector({0.1f, 0.2f});
    milvus::SearchResponse search_response;
    EXPECT_TRUE(session_a->Search(search_request, search_response).IsOk());

    EXPECT_CALL(service_, Query(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::QueryRequest* request,
                     milvus::proto::milvus::QueryResults*) {
            EXPECT_EQ(CountParam(request->query_params(), "cluster_id", "cluster-b"), 1);
            return ::grpc::Status{};
        });
    milvus::QueryRequest query_request;
    query_request.WithCollectionName("foo");
    milvus::QueryResponse query_response;
    EXPECT_TRUE(session_b->Query(query_request, query_response).IsOk());

    EXPECT_CALL(service_, HybridSearch(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::HybridSearchRequest* request,
                     milvus::proto::milvus::SearchResults* response) {
            EXPECT_EQ(CountParam(request->rank_params(), "cluster_id", "cluster-a"), 1);
            for (const auto& sub_request : request->requests()) {
                EXPECT_EQ(CountParam(sub_request.search_params(), "cluster_id", "cluster-a"), 0);
            }
            FillSearchResults(response);
            return ::grpc::Status{};
        });
    auto sub_request = std::make_shared<milvus::SubSearchRequest>();
    sub_request->WithLimit(1).AddFloatVector({0.1f, 0.2f});
    milvus::HybridSearchRequest hybrid_request;
    hybrid_request.WithCollectionName("foo")
        .AddSubRequest(sub_request)
        .WithLimit(1)
        .WithRerank(std::make_shared<milvus::RRFRerank>(60));
    milvus::HybridSearchResponse hybrid_response;
    EXPECT_TRUE(session_a->HybridSearch(hybrid_request, hybrid_response).IsOk());

    EXPECT_CALL(service_, Query(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::QueryRequest* request,
                     milvus::proto::milvus::QueryResults*) {
            EXPECT_EQ(CountParam(request->query_params(), "cluster_id", "cluster-a"), 0);
            return ::grpc::Status{};
        });
    EXPECT_TRUE(client->Query(query_request, query_response).IsOk());
}

TEST_F(UnconnectMilvusMockedTest, V2SessionIteratorsRouteEveryRequestWithoutMutatingInput) {
    auto client = CreateConnectedClient(service_, server_.ListenPort());
    milvus::MilvusClientV2SessionPtr session;
    ASSERT_TRUE(client->Session("cluster-a", session).IsOk());

    const std::string collection_name = "foo";
    milvus::CollectionSchema schema(collection_name);
    schema.AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "", true, false));
    schema.AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR).WithDimension(2));

    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .Times(2)
        .WillRepeatedly([&](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                            milvus::proto::milvus::DescribeCollectionResponse* response) {
            response->set_collectionid(100);
            milvus::ConvertCollectionSchema(schema, *response->mutable_schema());
            return ::grpc::Status{};
        });

    EXPECT_CALL(service_, Query(_, _, _))
        .Times(2)
        .WillRepeatedly([](::grpc::ServerContext*, const milvus::proto::milvus::QueryRequest* request,
                           milvus::proto::milvus::QueryResults* response) {
            EXPECT_EQ(CountParam(request->query_params(), "cluster_id", "cluster-a"), 1);
            response->set_session_ts(123);
            return ::grpc::Status{};
        });

    milvus::QueryIteratorRequest query_request;
    query_request.WithCollectionName(collection_name).WithLimit(1);
    milvus::QueryIteratorPtr query_iterator;
    EXPECT_TRUE(session->QueryIterator(query_request, query_iterator).IsOk());
    EXPECT_EQ(query_request.CollectionID(), 0);
    milvus::QueryResults query_results;
    EXPECT_TRUE(query_iterator->Next(query_results).IsOk());

    int search_calls = 0;
    EXPECT_CALL(service_, Search(_, _, _))
        .Times(2)
        .WillRepeatedly([&](::grpc::ServerContext*, const milvus::proto::milvus::SearchRequest* request,
                            milvus::proto::milvus::SearchResults* response) {
            EXPECT_EQ(CountParam(request->search_params(), "cluster_id", "cluster-a"), 1);
            ++search_calls;
            response->mutable_status()->set_code(milvus::proto::common::ErrorCode::Success);
            auto* results = response->mutable_results();
            results->set_num_queries(1);
            results->set_primary_field_name("id");
            results->mutable_search_iterator_v2_results()->set_token("token");
            if (search_calls == 1) {
                results->mutable_topks()->Add(0);
            } else {
                results->mutable_topks()->Add(1);
                results->mutable_ids()->mutable_int_id()->add_data(1);
                results->mutable_scores()->Add(0.1f);
            }
            return ::grpc::Status{};
        });

    milvus::SearchIteratorRequest search_request;
    search_request.WithCollectionName(collection_name).WithAnnsField("vec").WithMetricType(milvus::MetricType::COSINE);
    search_request.WithLimit(1).AddFloatVector({0.1f, 0.2f});
    milvus::SearchIteratorPtr search_iterator;
    EXPECT_TRUE(session->SearchIterator(search_request, search_iterator).IsOk());
    EXPECT_EQ(search_request.CollectionID(), 0);
    milvus::SingleResult search_results;
    EXPECT_TRUE(search_iterator->Next(search_results).IsOk());
}

TEST_F(UnconnectMilvusMockedTest, V2SessionSearchIteratorV1FallbackRoutesRequests) {
    auto client = CreateConnectedClient(service_, server_.ListenPort());
    milvus::MilvusClientV2SessionPtr session;
    ASSERT_TRUE(client->Session("cluster-a", session).IsOk());

    const std::string collection_name = "foo";
    milvus::CollectionSchema schema(collection_name);
    schema.AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "", true, false));
    schema.AddField(milvus::FieldSchema("vec", milvus::DataType::FLOAT_VECTOR).WithDimension(2));
    EXPECT_CALL(service_, DescribeCollection(_, _, _))
        .WillOnce([&](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                      milvus::proto::milvus::DescribeCollectionResponse* response) {
            response->set_collectionid(100);
            milvus::ConvertCollectionSchema(schema, *response->mutable_schema());
            return ::grpc::Status{};
        });

    int call_count = 0;
    EXPECT_CALL(service_, Search(_, _, _))
        .Times(2)
        .WillRepeatedly([&](::grpc::ServerContext*, const milvus::proto::milvus::SearchRequest* request,
                            milvus::proto::milvus::SearchResults* response) {
            EXPECT_EQ(CountParam(request->search_params(), "cluster_id", "cluster-a"), 1);
            ++call_count;
            response->mutable_status()->set_code(milvus::proto::common::ErrorCode::Success);
            auto* results = response->mutable_results();
            results->set_num_queries(1);
            results->set_primary_field_name("id");
            if (call_count == 1) {
                results->mutable_topks()->Add(0);
            } else {
                results->mutable_topks()->Add(1);
                results->mutable_ids()->mutable_int_id()->add_data(1);
                results->mutable_scores()->Add(0.1f);
            }
            return ::grpc::Status{};
        });

    milvus::SearchIteratorRequest request;
    request.WithCollectionName(collection_name).WithAnnsField("vec").WithMetricType(milvus::MetricType::COSINE);
    request.WithLimit(1).AddFloatVector({0.1f, 0.2f});
    milvus::SearchIteratorPtr iterator;
    EXPECT_TRUE(session->SearchIterator(request, iterator).IsOk());
    EXPECT_EQ(request.CollectionID(), 0);
}

TEST_F(UnconnectMilvusMockedTest, V2SessionGetRoutesTranslatedQuery) {
    auto client = CreateConnectedClient(service_, server_.ListenPort());
    milvus::MilvusClientV2SessionPtr session;
    ASSERT_TRUE(client->Session("cluster-a", session).IsOk());

    const std::string collection_name = "foo";
    milvus::CollectionSchema schema(collection_name);
    schema.AddField(milvus::FieldSchema("id", milvus::DataType::INT64, "", true, false));
    EXPECT_CALL(
        service_,
        DescribeCollection(
            _, Property(&milvus::proto::milvus::DescribeCollectionRequest::collection_name, collection_name), _))
        .WillOnce([&](::grpc::ServerContext*, const milvus::proto::milvus::DescribeCollectionRequest*,
                      milvus::proto::milvus::DescribeCollectionResponse* response) {
            response->set_collectionid(100);
            milvus::ConvertCollectionSchema(schema, *response->mutable_schema());
            return ::grpc::Status{};
        });
    EXPECT_CALL(service_, Query(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const milvus::proto::milvus::QueryRequest* request,
                     milvus::proto::milvus::QueryResults*) {
            EXPECT_EQ(CountParam(request->query_params(), "cluster_id", "cluster-a"), 1);
            return ::grpc::Status{};
        });

    milvus::GetRequest request;
    request.WithCollectionName(collection_name).WithIDs(std::vector<int64_t>{1});
    milvus::GetResponse response;
    EXPECT_TRUE(session->Get(request, response).IsOk());
}
