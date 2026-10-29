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

#include <algorithm>
#include <memory>
#include <utility>

#include "../mocks/MilvusMockedTest.h"
#include "milvus/MilvusClientV2.h"
#include "utils/Constants.h"

using ::milvus::proto::common::KeyValuePair;
using ::milvus::proto::milvus::DescribeCollectionRequest;
using ::milvus::proto::milvus::DescribeCollectionResponse;
using ::milvus::proto::milvus::QueryRequest;
using ::milvus::proto::milvus::QueryResults;
using ::testing::_;

namespace {

std::shared_ptr<milvus::MilvusClientV2>
CreateV2Client(testing::StrictMock<::milvus::MilvusMockedService>& service, uint16_t port) {
    EXPECT_CALL(service, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ::milvus::proto::milvus::ConnectRequest*,
                     ::milvus::proto::milvus::ConnectResponse*) { return ::grpc::Status{}; });

    auto client = milvus::MilvusClientV2::Create();
    milvus::ConnectParam connect_param{"127.0.0.1", port};
    auto status = client->Connect(connect_param);
    EXPECT_TRUE(status.IsOk());
    return client;
}

void
MockDescribeCollection(::milvus::MilvusMockedService& service, const std::string& collection_name) {
    EXPECT_CALL(service, DescribeCollection(_, _, _))
        .WillOnce([&](::grpc::ServerContext*, const DescribeCollectionRequest* request,
                      DescribeCollectionResponse* response) {
            EXPECT_EQ(request->collection_name(), collection_name);
            response->set_collectionid(100);
            auto* schema = response->mutable_schema();
            schema->set_name(collection_name);
            auto* pk_field = schema->add_fields();
            pk_field->set_name("id");
            pk_field->set_is_primary_key(true);
            pk_field->set_data_type(::milvus::proto::schema::DataType::Int64);
            auto* tags_field = schema->add_fields();
            tags_field->set_name("tags");
            tags_field->set_data_type(::milvus::proto::schema::DataType::Array);
            return ::grpc::Status{};
        });
}

::milvus::proto::schema::FieldData
MakeInt64Field(const std::string& name, std::vector<int64_t> values) {
    ::milvus::proto::schema::FieldData field;
    field.set_field_name(name);
    field.set_type(::milvus::proto::schema::DataType::Int64);
    auto* scalar = field.mutable_scalars();
    for (auto value : values) {
        scalar->mutable_long_data()->add_data(value);
    }
    return field;
}

::milvus::proto::schema::FieldData
MakeVarCharField(const std::string& name, std::vector<std::string> values) {
    ::milvus::proto::schema::FieldData field;
    field.set_field_name(name);
    field.set_type(::milvus::proto::schema::DataType::VarChar);
    auto* scalar = field.mutable_scalars();
    for (auto& value : values) {
        scalar->mutable_string_data()->add_data(value);
    }
    return field;
}

// The session-setup probe uses limit=1 and returns no data page; every data page carries one
// entity per requested element (the server caps element_filter pages by element count).
void
MockElementFilterQuery(::milvus::MilvusMockedService& service) {
    EXPECT_CALL(service, Query(_, _, _))
        .WillRepeatedly([&](::grpc::ServerContext*, const QueryRequest* request, QueryResults* response) {
            uint64_t session_ts = 999999;
            int64_t query_limit = 0;
            for (const auto& pair : request->query_params()) {
                if (pair.key() == milvus::LIMIT) {
                    query_limit = std::atoi(pair.value().c_str());
                }
            }
            if (query_limit == 1 && request->output_fields_size() == 0) {
                // the session-setup probe query
                response->set_session_ts(session_ts);
                return ::grpc::Status{};
            }

            // derive the cursor from the inclusive or exclusive pk predicate in the expr
            int64_t start = 0;
            const std::string& expr = request->expr();
            auto inclusive = expr.find("id >= ");
            auto exclusive = expr.find("id > ");
            if (inclusive != std::string::npos) {
                start = std::atoll(expr.c_str() + inclusive + 6) + 1;
            } else if (exclusive != std::string::npos) {
                start = std::atoll(expr.c_str() + exclusive + 5) + 1;
            }

            int64_t count = query_limit > 0 ? query_limit : 1;
            std::vector<int64_t> ids;
            for (auto i = 0; i < count && start + i < 4; ++i) {
                ids.push_back(start + i);
            }
            response->set_session_ts(session_ts);
            response->mutable_fields_data()->Add(MakeInt64Field("id", ids));
            response->add_output_fields("id");
            for (auto id : ids) {
                auto* indices = response->add_element_indices();
                indices->mutable_indices()->add_data(id);
            }
            return ::grpc::Status{};
        });
}

// Each entity carries two matched elements, and the server caps a page by element count
// (elementLimit = limit). With batch_size=2 a single entity's matched elements are split
// across two pages, so the resume must re-enter the entity and return only the remaining
// element instead of skipping the entity entirely.
void
MockElementFilterMultiElementQuery(::milvus::MilvusMockedService& service) {
    EXPECT_CALL(service, Query(_, _, _))
        .WillRepeatedly([&](::grpc::ServerContext*, const QueryRequest* request, QueryResults* response) {
            uint64_t session_ts = 999999;
            int64_t query_limit = 0;
            for (const auto& pair : request->query_params()) {
                if (pair.key() == milvus::LIMIT) {
                    query_limit = std::atoi(pair.value().c_str());
                }
            }
            if (query_limit == 1 && request->output_fields_size() == 0) {
                response->set_session_ts(session_ts);
                return ::grpc::Status{};
            }

            // element-filter resume re-enters the last pk with >= plus the element offset;
            // plain seek uses a strict > and omits the element params
            int64_t start = 0;
            int64_t reenter_offset = 0;
            bool has_element_params = false;
            const std::string& expr = request->expr();
            auto inclusive = expr.find("id >= ");
            auto exclusive = expr.find("id > ");
            for (const auto& pair : request->query_params()) {
                if (pair.key() == milvus::QUERY_ITER_LAST_PK || pair.key() == milvus::QUERY_ITER_LAST_ELEMENT_OFFSET) {
                    has_element_params = true;
                }
            }
            if (has_element_params) {
                // resumed within an entity: start at the same pk, continue after the offset
                if (inclusive != std::string::npos) {
                    start = std::atoll(expr.c_str() + inclusive + 6);
                }
                for (const auto& pair : request->query_params()) {
                    if (pair.key() == milvus::QUERY_ITER_LAST_ELEMENT_OFFSET) {
                        reenter_offset = std::atoll(pair.value().c_str()) + 1;
                    }
                }
            } else if (inclusive != std::string::npos) {
                start = std::atoll(expr.c_str() + inclusive + 6) + 1;
            } else if (exclusive != std::string::npos) {
                start = std::atoll(expr.c_str() + exclusive + 5) + 1;
            }

            // cap the page by element count: two elements per entity
            int64_t entities = (query_limit + 1) / 2;
            std::vector<int64_t> ids;
            std::vector<std::vector<int64_t>> index_lists;
            for (auto i = 0; i < entities && start + i < 4; ++i) {
                ids.push_back(start + i);
                index_lists.push_back({reenter_offset, reenter_offset + 1});
            }
            response->set_session_ts(session_ts);
            response->mutable_fields_data()->Add(MakeInt64Field("id", ids));
            response->add_output_fields("id");
            for (const auto& indices : index_lists) {
                auto* element_indices = response->add_element_indices();
                for (auto index : indices) {
                    element_indices->mutable_indices()->add_data(index);
                }
            }
            return ::grpc::Status{};
        });
}

void
MockDescribeVarCharCollection(::milvus::MilvusMockedService& service, const std::string& collection_name) {
    EXPECT_CALL(service, DescribeCollection(_, _, _))
        .WillOnce([&](::grpc::ServerContext*, const DescribeCollectionRequest* request,
                      DescribeCollectionResponse* response) {
            EXPECT_EQ(request->collection_name(), collection_name);
            response->set_collectionid(101);
            auto* schema = response->mutable_schema();
            schema->set_name(collection_name);
            auto* pk_field = schema->add_fields();
            pk_field->set_name("title");
            pk_field->set_is_primary_key(true);
            pk_field->set_data_type(::milvus::proto::schema::DataType::VarChar);
            auto* tags_field = schema->add_fields();
            tags_field->set_name("tags");
            tags_field->set_data_type(::milvus::proto::schema::DataType::Array);
            return ::grpc::Status{};
        });
}

// A single entity with two matched elements, one per page (element-filter resume).
void
MockVarCharElementFilterQuery(::milvus::MilvusMockedService& service) {
    EXPECT_CALL(service, Query(_, _, _))
        .WillRepeatedly([&](::grpc::ServerContext*, const QueryRequest* request, QueryResults* response) {
            uint64_t session_ts = 999999;
            int64_t query_limit = 0;
            for (const auto& pair : request->query_params()) {
                if (pair.key() == milvus::LIMIT) {
                    query_limit = std::atoi(pair.value().c_str());
                }
            }
            if (query_limit == 1 && request->output_fields_size() == 0) {
                response->set_session_ts(session_ts);
                return ::grpc::Status{};
            }

            std::string start = "row-0";
            int64_t reenter_offset = 0;
            bool has_element_params = false;
            const std::string& expr = request->expr();
            auto inclusive = expr.find("title >= ");
            auto exclusive = expr.find("title > ");
            for (const auto& pair : request->query_params()) {
                if (pair.key() == milvus::QUERY_ITER_LAST_PK || pair.key() == milvus::QUERY_ITER_LAST_ELEMENT_OFFSET) {
                    has_element_params = true;
                }
            }
            if (has_element_params) {
                for (const auto& pair : request->query_params()) {
                    if (pair.key() == milvus::QUERY_ITER_LAST_ELEMENT_OFFSET) {
                        reenter_offset = std::atoll(pair.value().c_str()) + 1;
                    }
                }
            } else if (inclusive != std::string::npos) {
                auto begin = expr.c_str() + inclusive + 8;
                auto end = begin;
                while (*end != '\0' && *end != ' ') {
                    ++end;
                }
                start.assign(begin, end);
            } else if (exclusive != std::string::npos) {
                auto begin = expr.c_str() + exclusive + 7;
                auto end = begin;
                while (*end != '\0' && *end != ' ') {
                    ++end;
                }
                start.assign(begin, end);
            }

            response->set_session_ts(session_ts);
            response->mutable_fields_data()->Add(MakeVarCharField("title", {start}));
            response->add_output_fields("title");
            auto* element_indices = response->add_element_indices();
            element_indices->mutable_indices()->add_data(reenter_offset);
            element_indices->mutable_indices()->add_data(reenter_offset + 1);
            return ::grpc::Status{};
        });
}

}  // namespace

TEST_F(UnconnectMilvusMockedTest, V2QueryIteratorElementFilterResume) {
    auto client = CreateV2Client(service_, server_.ListenPort());
    const std::string collection_name = "element_filter_collection";
    MockDescribeCollection(service_, collection_name);
    MockElementFilterQuery(service_);

    milvus::QueryIteratorRequest request;
    request.SetCollectionName(collection_name);
    request.SetBatchSize(1);
    request.SetFilter("element_filter(tags, tag == \"x\") and mock_element_filter");
    request.AddOutputField("id");
    milvus::QueryIteratorPtr iterator;
    auto status = client->QueryIterator(request, iterator);
    ASSERT_TRUE(status.IsOk());

    milvus::QueryResults results;
    status = iterator->Next(results);
    ASSERT_TRUE(status.IsOk());
    ASSERT_EQ(results.GetRowCount(), 1);

    // the element-filter cursor must carry the element position within the last primary key
    milvus::QueryIteratorCursor cursor;
    status = iterator->Cursor(cursor);
    ASSERT_TRUE(status.IsOk());
    ASSERT_TRUE(cursor.HasIntPk());
    ASSERT_TRUE(cursor.HasLastElementOffset());

    // resume a new iterator from the captured cursor: the first data page must re-enter the
    // cursor primary key with an inclusive predicate and the query_iter_last_* params
    milvus::QueryIteratorRequest resume_request;
    resume_request.SetCollectionName(collection_name);
    resume_request.SetBatchSize(1);
    resume_request.SetFilter("element_filter(tags, tag == \"x\") and mock_element_filter");
    resume_request.AddOutputField("id");
    resume_request.WithCursor(cursor);
    milvus::QueryIteratorPtr resumed;
    status = client->QueryIterator(resume_request, resumed);
    ASSERT_TRUE(status.IsOk());

    milvus::QueryResults resumed_results;
    status = resumed->Next(resumed_results);
    ASSERT_TRUE(status.IsOk());
    ASSERT_EQ(resumed_results.GetRowCount(), 1);
}

TEST_F(UnconnectMilvusMockedTest, V2QueryIteratorElementFilterSeekOmitsElementParams) {
    auto client = CreateV2Client(service_, server_.ListenPort());
    const std::string collection_name = "element_filter_collection";
    MockDescribeCollection(service_, collection_name);

    // the seek path uses iterator=false and must NOT carry the element-resume params
    // (Milvus rejects query_iter_last_* unless iterator=true)
    int call_count = 0;
    EXPECT_CALL(service_, Query(_, _, _))
        .WillRepeatedly([&](::grpc::ServerContext*, const QueryRequest* request, QueryResults* response) {
            ++call_count;
            response->set_session_ts(999999);
            if (call_count == 1) {
                // session-setup probe
                return ::grpc::Status{};
            }
            bool is_seek = false;
            for (const auto& pair : request->query_params()) {
                if (pair.key() == milvus::ITERATOR_FIELD && pair.value() == "False") {
                    is_seek = true;
                }
            }
            if (is_seek) {
                for (const auto& pair : request->query_params()) {
                    EXPECT_NE(pair.key(), milvus::QUERY_ITER_LAST_PK);
                    EXPECT_NE(pair.key(), milvus::QUERY_ITER_LAST_ELEMENT_OFFSET);
                }
                response->mutable_fields_data()->Add(MakeInt64Field("id", {1}));
                response->add_element_indices()->mutable_indices()->add_data(1);
            } else {
                bool has_element_params = false;
                for (const auto& pair : request->query_params()) {
                    if (pair.key() == milvus::QUERY_ITER_LAST_PK ||
                        pair.key() == milvus::QUERY_ITER_LAST_ELEMENT_OFFSET) {
                        has_element_params = true;
                    }
                }
                EXPECT_TRUE(has_element_params);
                response->mutable_fields_data()->Add(MakeInt64Field("id", {2}));
                response->add_element_indices()->mutable_indices()->add_data(2);
            }
            return ::grpc::Status{};
        });

    milvus::QueryIteratorRequest request;
    request.SetCollectionName(collection_name);
    request.SetBatchSize(1);
    request.SetOffset(1);
    request.SetFilter("element_filter(tags, tag == \"x\") and mock_element_filter");
    request.AddOutputField("id");
    milvus::QueryIteratorPtr iterator;
    auto status = client->QueryIterator(request, iterator);
    ASSERT_TRUE(status.IsOk());

    milvus::QueryResults results;
    status = iterator->Next(results);
    ASSERT_TRUE(status.IsOk());
    ASSERT_EQ(results.GetRowCount(), 1);
}

// A single entity with two matched elements must not terminate the iterator after the first
// page: the server caps the page by element count, so the resume re-enters the entity and
// returns only the remaining element.
TEST_F(UnconnectMilvusMockedTest, V2QueryIteratorElementFilterResumesWithinEntity) {
    auto client = CreateV2Client(service_, server_.ListenPort());
    const std::string collection_name = "element_filter_collection";
    MockDescribeCollection(service_, collection_name);
    MockElementFilterMultiElementQuery(service_);

    milvus::QueryIteratorRequest request;
    request.SetCollectionName(collection_name);
    request.SetBatchSize(2);
    request.SetFilter("element_filter(tags, tag == \"x\") and multi_element_mock");
    request.AddOutputField("id");
    milvus::QueryIteratorPtr iterator;
    auto status = client->QueryIterator(request, iterator);
    ASSERT_TRUE(status.IsOk());

    // first page: elementLimit=2 caps it to one entity with elements [0,1]
    milvus::QueryResults results;
    status = iterator->Next(results);
    ASSERT_TRUE(status.IsOk());
    ASSERT_EQ(results.GetRowCount(), 1);
    ASSERT_EQ(results.ElementIndices().size(), 1);
    ASSERT_EQ(results.ElementIndices()[0], (std::vector<int64_t>{0, 1}));

    // the cursor must capture the element position within the last primary key
    milvus::QueryIteratorCursor cursor;
    status = iterator->Cursor(cursor);
    ASSERT_TRUE(status.IsOk());
    ASSERT_TRUE(cursor.HasIntPk());
    ASSERT_TRUE(cursor.HasLastElementOffset());
    ASSERT_EQ(cursor.LastElementOffset(), 1);

    // resume re-enters primary key 0 at element offset 2 (elements 0,1 were already returned)
    milvus::QueryIteratorRequest resume_request;
    resume_request.SetCollectionName(collection_name);
    resume_request.SetBatchSize(2);
    resume_request.SetFilter("element_filter(tags, tag == \"x\") and multi_element_mock");
    resume_request.AddOutputField("id");
    resume_request.WithCursor(cursor);
    milvus::QueryIteratorPtr resumed;
    status = client->QueryIterator(resume_request, resumed);
    ASSERT_TRUE(status.IsOk());

    milvus::QueryResults resumed_results;
    status = resumed->Next(resumed_results);
    ASSERT_TRUE(status.IsOk());
    ASSERT_EQ(resumed_results.GetRowCount(), 1);
    ASSERT_EQ(resumed_results.ElementIndices().size(), 1);
    ASSERT_EQ(resumed_results.ElementIndices()[0], (std::vector<int64_t>{2, 3}));
}

// The element-filter resume must carry an inclusive VarChar pk predicate and the raw pk in
// query_iter_last_pk (not the JSON-quoted form).
TEST_F(UnconnectMilvusMockedTest, V2QueryIteratorVarCharElementFilterResume) {
    auto client = CreateV2Client(service_, server_.ListenPort());
    const std::string collection_name = "varchar_element_filter_collection";
    MockDescribeVarCharCollection(service_, collection_name);
    MockVarCharElementFilterQuery(service_);

    milvus::QueryIteratorRequest request;
    request.SetCollectionName(collection_name);
    request.SetBatchSize(1);
    request.SetFilter("element_filter(tags, tag == \"x\") and varchar_element_mock");
    request.AddOutputField("title");
    milvus::QueryIteratorPtr iterator;
    auto status = client->QueryIterator(request, iterator);
    ASSERT_TRUE(status.IsOk());

    milvus::QueryResults results;
    status = iterator->Next(results);
    ASSERT_TRUE(status.IsOk());
    ASSERT_EQ(results.GetRowCount(), 1);

    milvus::QueryIteratorCursor cursor;
    status = iterator->Cursor(cursor);
    ASSERT_TRUE(status.IsOk());
    ASSERT_TRUE(cursor.HasStrPk());
    ASSERT_EQ(cursor.StrPk(), "row-0");
    ASSERT_TRUE(cursor.HasLastElementOffset());

    milvus::QueryIteratorRequest resume_request;
    resume_request.SetCollectionName(collection_name);
    resume_request.SetBatchSize(1);
    resume_request.SetFilter("element_filter(tags, tag == \"x\") and varchar_element_mock");
    resume_request.AddOutputField("title");
    resume_request.WithCursor(cursor);
    milvus::QueryIteratorPtr resumed;
    status = client->QueryIterator(resume_request, resumed);
    ASSERT_TRUE(status.IsOk());

    milvus::QueryResults resumed_results;
    status = resumed->Next(resumed_results);
    ASSERT_TRUE(status.IsOk());
    ASSERT_EQ(resumed_results.GetRowCount(), 1);
}

// A cursor cannot be captured before any page is delivered: there is no primary-key position
// to resume from, so the capture must fail with a clear error instead of a pk-less cursor.
TEST_F(UnconnectMilvusMockedTest, V2QueryIteratorCursorRejectsEmptyCapture) {
    auto client = CreateV2Client(service_, server_.ListenPort());
    const std::string collection_name = "element_filter_collection";
    MockDescribeCollection(service_, collection_name);

    // the session-setup probe only (no data page is delivered)
    EXPECT_CALL(service_, Query(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const QueryRequest* request, QueryResults* response) {
            response->set_session_ts(999999);
            return ::grpc::Status{};
        });

    milvus::QueryIteratorRequest request;
    request.SetCollectionName(collection_name);
    request.SetBatchSize(1);
    request.SetFilter("element_filter(tags, tag == \"x\") and mock_element_filter");
    request.AddOutputField("id");
    milvus::QueryIteratorPtr iterator;
    auto status = client->QueryIterator(request, iterator);
    ASSERT_TRUE(status.IsOk());

    milvus::QueryIteratorCursor cursor;
    status = iterator->Cursor(cursor);
    ASSERT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
    EXPECT_NE(status.Message().find("no primary key"), std::string::npos);
}

// The element-filter offset seek must consume the offset in matched-element terms, not entity
// rows: an entity carrying two matched elements advances the seek by two.
TEST_F(UnconnectMilvusMockedTest, V2QueryIteratorElementFilterSeekCountsElements) {
    auto client = CreateV2Client(service_, server_.ListenPort());
    const std::string collection_name = "element_filter_collection";
    MockDescribeCollection(service_, collection_name);

    int call_count = 0;
    std::vector<int64_t> seek_limits;
    EXPECT_CALL(service_, Query(_, _, _))
        .WillRepeatedly([&](::grpc::ServerContext*, const QueryRequest* request, QueryResults* response) {
            ++call_count;
            response->set_session_ts(999999);
            if (call_count == 1) {
                // session-setup probe
                return ::grpc::Status{};
            }
            bool is_seek = false;
            for (const auto& pair : request->query_params()) {
                if (pair.key() == milvus::ITERATOR_FIELD && pair.value() == "False") {
                    is_seek = true;
                } else if (pair.key() == milvus::LIMIT) {
                    seek_limits.push_back(std::atoi(pair.value().c_str()));
                }
            }
            if (is_seek) {
                // each seek page carries one entity with two matched elements, so a seek of 4
                // consumes two pages of 2 elements each
                response->mutable_fields_data()->Add(MakeInt64Field("id", {1}));
                auto* indices = response->add_element_indices();
                indices->mutable_indices()->add_data(1);
                indices->mutable_indices()->add_data(2);
            } else {
                response->mutable_fields_data()->Add(MakeInt64Field("id", {2}));
                response->add_element_indices()->mutable_indices()->add_data(2);
            }
            return ::grpc::Status{};
        });

    milvus::QueryIteratorRequest request;
    request.SetCollectionName(collection_name);
    request.SetBatchSize(1);
    request.SetOffset(4);
    request.SetFilter("element_filter(tags, tag == \"x\") and mock_element_filter");
    request.AddOutputField("id");
    milvus::QueryIteratorPtr iterator;
    auto status = client->QueryIterator(request, iterator);
    ASSERT_TRUE(status.IsOk());

    milvus::QueryResults results;
    status = iterator->Next(results);
    ASSERT_TRUE(status.IsOk());
    ASSERT_EQ(results.GetRowCount(), 1);
    // the seek consumed offset=4 in two pages: the first page requests limit=4 and yields
    // 2 elements, the second requests limit=2 and yields 2 elements, leaving offset=0
    // (element-granular, not entity-granular). A probe/data query records limit=1 too.
    EXPECT_EQ(seek_limits.size(), 3);
    std::sort(seek_limits.begin(), seek_limits.end());
    EXPECT_EQ(seek_limits, (std::vector<int64_t>{1, 2, 4}));
}

// When a finite Limit truncates a batch, the captured cursor must resume after the last
// delivered row, not after the whole untrimmed batch.
TEST_F(UnconnectMilvusMockedTest, V2QueryIteratorCursorAfterLimitTrimsBatch) {
    auto client = CreateV2Client(service_, server_.ListenPort());
    const std::string collection_name = "element_filter_collection";
    MockDescribeCollection(service_, collection_name);

    int data_calls = 0;
    EXPECT_CALL(service_, Query(_, _, _))
        .WillRepeatedly([&](::grpc::ServerContext*, const QueryRequest* request, QueryResults* response) {
            uint64_t session_ts = 999999;
            int64_t query_limit = 0;
            for (const auto& pair : request->query_params()) {
                if (pair.key() == milvus::LIMIT) {
                    query_limit = std::atoi(pair.value().c_str());
                }
            }
            if (query_limit == 1 && request->output_fields_size() == 0) {
                response->set_session_ts(session_ts);
                return ::grpc::Status{};
            }
            ++data_calls;
            response->set_session_ts(session_ts);
            response->mutable_fields_data()->Add(MakeInt64Field("id", {1, 2, 3}));
            response->add_output_fields("id");
            for (auto id : {1, 2, 3}) {
                auto* indices = response->add_element_indices();
                indices->mutable_indices()->add_data(id);
            }
            return ::grpc::Status{};
        });

    // limit=1 with a batch that returns 3 rows: only the first row is delivered
    milvus::QueryIteratorRequest request;
    request.SetCollectionName(collection_name);
    request.SetBatchSize(100);
    request.SetLimit(1);
    request.SetFilter("element_filter(tags, tag == \"x\") and mock_element_filter");
    request.AddOutputField("id");
    milvus::QueryIteratorPtr iterator;
    auto status = client->QueryIterator(request, iterator);
    ASSERT_TRUE(status.IsOk());

    milvus::QueryResults results;
    status = iterator->Next(results);
    ASSERT_TRUE(status.IsOk());
    ASSERT_EQ(results.GetRowCount(), 1);

    milvus::QueryIteratorCursor cursor;
    status = iterator->Cursor(cursor);
    ASSERT_TRUE(status.IsOk());
    ASSERT_TRUE(cursor.HasIntPk());
    // the cursor must point at the last delivered row (pk 1), not the end of the batch (pk 3)
    EXPECT_EQ(cursor.IntPk(), 1);
}
