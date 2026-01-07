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

#include <iostream>
#include <sstream>
#include <thread>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {
const char* const collection_name = "CPP_V2_DML";
const char* const field_id = "pk";
const char* const field_vector = "vector";
const char* const field_text = "text";
const uint32_t dimension = 4;

void
printRowCount(milvus::MilvusClientV2Ptr& client, milvus::ConsistencyLevel level) {
    auto request = milvus::QueryRequest()
                       .WithCollectionName(collection_name)
                       .AddOutputField("count(*)")
                       .WithConsistencyLevel(level);

    milvus::QueryResponse response;
    auto status = client->Query(request, response);
    util::CheckStatus("query count(*)", status);
    std::cout << "count(*) = " << response.Results().GetRowCount() << std::endl;
}

std::vector<int64_t>
buildCollection(milvus::MilvusClientV2Ptr& client, bool auto_id) {
    // collection schema, drop and create collection
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->AddField({field_id, milvus::DataType::INT64, "id", true, auto_id});
    collection_schema->AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
    collection_schema->AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR).WithMaxLength(100));

    auto status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + std::string(collection_name), status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::HNSW, milvus::MetricType::L2);
    index_vector.AddExtraParam("M", "64");
    index_vector.AddExtraParam("efConstruction", "200");
    milvus::IndexDesc index_text(field_text, "", milvus::IndexType::INVERTED);
    status = client->CreateIndex(milvus::CreateIndexRequest()
                                     .WithCollectionName(collection_name)
                                     .AddIndex(std::move(index_vector))
                                     .AddIndex(std::move(index_text)));
    util::CheckStatus("create indexes on collection", status);

    // load collection
    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection: " + std::string(collection_name), status);

    {
        // insert somes rows by column-based
        auto texts = std::vector<std::string>{"column-based-1", "column-based-2"};
        auto vectors = util::GenerateFloatVectors(dimension, 2);
        std::vector<milvus::FieldDataPtr> fields_data{
            std::make_shared<milvus::VarCharFieldData>(field_text, texts),
            std::make_shared<milvus::FloatVecFieldData>(field_vector, vectors)};

        if (!auto_id) {
            auto ids = std::vector<int64_t>{10000, 10001};
            fields_data.emplace_back(std::make_shared<milvus::Int64FieldData>(field_id, ids));
        }

        milvus::InsertResponse resp_insert;
        status = client->Insert(
            milvus::InsertRequest().WithCollectionName(collection_name).WithColumnsData(std::move(fields_data)),
            resp_insert);
        util::CheckStatus("insert", status);
        std::cout << resp_insert.Results().InsertCount() << " rows inserted by column-based." << std::endl;
    }

    // insert some rows by row-based
    const int64_t row_count = 100;
    milvus::EntityRows rows;
    for (auto i = 0; i < row_count; ++i) {
        milvus::EntityRow row;
        if (!auto_id) {
            row[field_id] = i;
        }
        row[field_text] = "hello world " + std::to_string(i);
        row[field_vector] = util::GenerateFloatVector(dimension);
        rows.emplace_back(std::move(row));
    }

    milvus::InsertResponse resp_insert;
    status = client->Insert(milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)),
                            resp_insert);
    util::CheckStatus("insert", status);
    std::cout << resp_insert.Results().InsertCount() << " rows inserted by row-based." << std::endl;

    // get row count, 102 rows at this point
    printRowCount(client, milvus::ConsistencyLevel::STRONG);

    // if auto_id is true, the auto-generated ids are returned by the resp_insert
    // else the user-defined ids are returned by the resp_insert
    return resp_insert.Results().IdArray().IntIDArray();
}

std::string
combineFilterExpr(const std::vector<int64_t>& ids) {
    std::stringstream ss;
    ss << std::string(field_id) << " in [";
    if (!ids.empty()) {
        ss << ids[0];
        for (size_t i = 1; i < ids.size(); ++i) {
            ss << "," << ids[i];
        }
    }
    ss << "]";
    return ss.str();
}

void
query(milvus::MilvusClientV2Ptr& client, const std::string& filter, milvus::ConsistencyLevel level) {
    auto request = milvus::QueryRequest()
                       .WithCollectionName(collection_name)
                       .WithFilter(filter)
                       .AddOutputField(field_id)
                       .AddOutputField(field_text)
                       .AddOutputField(field_vector)
                       .WithConsistencyLevel(level);

    std::cout << "Query with expression: " << filter << std::endl;
    milvus::QueryResponse response;
    auto status = client->Query(request, response);
    util::CheckStatus("query", status);

    std::cout << "Query result count: " << std::to_string(response.Results().GetRowCount()) << std::endl;
    const auto& query_results = response.Results();
    for (auto i = 0; i < query_results.GetRowCount(); i++) {
        milvus::EntityRow output_row;
        status = query_results.OutputRow(i, output_row);
        util::CheckStatus("get output row", status);
        std::cout << "\t" << output_row << std::endl;
    }
}

void
doDml(milvus::MilvusClientV2Ptr& client, bool auto_id) {
    std::cout << "\n================== auto_id: " << (auto_id ? "true" : "false") << " ==================" << std::endl;
    auto ids = buildCollection(client, auto_id);

    // upsert some rows
    int64_t old_id_1 = ids[1];
    int64_t old_id_2 = ids[ids.size() - 1];
    milvus::EntityRows upsert_rows;
    std::vector<float> dummy_vector(dimension);
    for (auto d = 0; d < dimension; ++d) {
        dummy_vector[d] = 0.88;
    }
    {
        milvus::EntityRow row;
        row[field_id] = old_id_1;
        row[field_text] = "this row is updated from " + std::to_string(old_id_1);
        row[field_vector] = dummy_vector;
        upsert_rows.emplace_back(std::move(row));
    }
    {
        milvus::EntityRow row;
        row[field_id] = old_id_2;
        row[field_text] = "this row is updated from " + std::to_string(old_id_2);
        row[field_vector] = dummy_vector;
        upsert_rows.emplace_back(std::move(row));
    }

    milvus::UpsertResponse resp_upsert;
    auto status = client->Upsert(
        milvus::UpsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(upsert_rows)), resp_upsert);
    util::CheckStatus("upsert", status);

    // if the primary key is auto-id, upsert() will delete the old id and create a new id,
    // this behavior is a technical trade-off of milvus
    const auto new_ids = resp_upsert.Results().IdArray().IntIDArray();
    std::cout << "After upsert, the id " << old_id_1 << " has been updated to " << new_ids[0] << std::endl;
    std::cout << "After upsert, the id " << old_id_2 << " has been updated to " << new_ids[1] << std::endl;

    // query the updated items
    // the SESSION level ensures that the previous dml change of this process must be
    // visible to the next query/search of the same process.
    std::string filter = combineFilterExpr(new_ids);
    query(client, filter, milvus::ConsistencyLevel::SESSION);

    // get row count, 102 rows at this point
    // the previous query has used SESSION to ensure the data is upsert action is consumed,
    // we can use EVENTUALLY level to ignore dml consistency check(in the server side)
    printRowCount(client, milvus::ConsistencyLevel::EVENTUALLY);

    // partial update the two items
    // note that in partial update, we dont' need to input all fields, the vector field is ignored
    milvus::EntityRows partial_upsert_rows = {{{field_id, new_ids[0]}, {field_text, "this item is partial updated"}},
                                              {{field_id, new_ids[1]}, {field_text, "this item is partial updated"}}};
    status = client->Upsert(milvus::UpsertRequest()
                                .WithCollectionName(collection_name)
                                .WithRowsData(std::move(partial_upsert_rows))
                                .WithPartialUpdate(true),
                            resp_upsert);
    util::CheckStatus("partial upsert", status);

    const auto updated_ids = resp_upsert.Results().IdArray().IntIDArray();
    std::cout << "After partial upsert, the id " << new_ids[0] << " has been updated to " << updated_ids[0]
              << std::endl;
    std::cout << "After partial upsert, the id " << new_ids[1] << " has been updated to " << updated_ids[1]
              << std::endl;

    // query the updated items
    // the SESSION level ensures that the previous dml change of this process must be
    // visible to the next query/search of the same process.
    filter = combineFilterExpr(updated_ids);
    query(client, filter, milvus::ConsistencyLevel::SESSION);

    // get row count, 102 rows at this point
    // the previous query has used SESSION to ensure the data is upsert action is consumed,
    // we can use EVENTUALLY level to ignore dml consistency check(in the server side)
    printRowCount(client, milvus::ConsistencyLevel::EVENTUALLY);

    // delete the two items
    std::cout << "Delete with expression: " << filter << std::endl;
    milvus::DeleteResponse resp_delete;
    status =
        client->Delete(milvus::DeleteRequest().WithCollectionName(collection_name).WithFilter(filter), resp_delete);
    util::CheckStatus("delete", status);

    // query immediatelly again with STRONG level, result must be empty.
    // set to STRONG level so that the query is executed after the delete action is consumed by server
    query(client, filter, milvus::ConsistencyLevel::SESSION);

    // get the numer of rows after delete, must be 102 - 2 = 100
    // no data changed after the last query, we can use EVENTUALLY level to ignore
    // dml consistency check(in the server side)
    printRowCount(client, milvus::ConsistencyLevel::EVENTUALLY);
}

}  // namespace

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClientV2::Create();

    milvus::ConnectParam connect_param{"localhost", 19530, "root", "Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    // primary keys are auto-generated by milvus server
    doDml(client, true);

    // primary keys are user-defined
    doDml(client, false);

    client->Disconnect();
    return 0;
}
