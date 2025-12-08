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
#include <string>
#include <thread>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClientV2::Create();

    milvus::ConnectParam connect_param{"localhost", 19530, "root", "Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    const std::string collection_name = "CPP_V2_DML";
    const std::string field_id = "pk";
    const std::string field_vector = "vector";
    const std::string field_text = "text";
    const uint32_t dimension = 4;

    // collection schema, drop and create collection
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>(collection_name);
    collection_schema->AddField({field_id, milvus::DataType::INT64, "id", true, true});
    collection_schema->AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
    collection_schema->AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR).WithMaxLength(100));

    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(milvus::CreateCollectionRequest().WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + collection_name, status);

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
    util::CheckStatus("load collection: " + collection_name, status);

    {
        // insert somes rows by column-based
        auto texts = std::vector<std::string>{"column-based-1", "column-based-2"};
        auto vectors = util::GenerateFloatVectors(dimension, 2);
        std::vector<milvus::FieldDataPtr> fields_data{
            std::make_shared<milvus::VarCharFieldData>(field_text, texts),
            std::make_shared<milvus::FloatVecFieldData>(field_vector, vectors)};
        milvus::InsertResponse resp_insert;
        status = client->Insert(
            milvus::InsertRequest().WithCollectionName(collection_name).WithColumnsData(std::move(fields_data)),
            resp_insert);
        util::CheckStatus("insert", status);
        std::cout << resp_insert.Results().InsertCount() << " rows inserted by column-based." << std::endl;
    }

    // insert some rows
    const int64_t row_count = 100;
    milvus::EntityRows rows;
    for (auto i = 0; i < row_count; ++i) {
        milvus::EntityRow row;
        row[field_text] = "hello world " + std::to_string(i);
        row[field_vector] = util::GenerateFloatVector(dimension);
        rows.emplace_back(std::move(row));
    }

    milvus::InsertResponse resp_insert;
    status = client->Insert(milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)),
                            resp_insert);
    util::CheckStatus("insert", status);
    std::cout << resp_insert.Results().InsertCount() << " rows inserted by row-based." << std::endl;
    const auto& ids = resp_insert.Results().IdArray().IntIDArray();

    // upsert some rows
    int64_t update_id_1 = ids[1];
    int64_t update_id_2 = ids[ids.size() - 1];
    milvus::EntityRows upsert_rows;
    std::vector<float> dummy_vector(dimension);
    for (auto d = 0; d < dimension; ++d) {
        dummy_vector[d] = 0.88;
    }
    {
        milvus::EntityRow row;
        row[field_id] = update_id_1;
        row[field_text] = "this row is updated from " + std::to_string(update_id_1);
        row[field_vector] = dummy_vector;
        upsert_rows.emplace_back(std::move(row));
    }
    {
        milvus::EntityRow row;
        row[field_id] = update_id_2;
        row[field_text] = "this row is updated from " + std::to_string(update_id_2);
        row[field_vector] = dummy_vector;
        upsert_rows.emplace_back(std::move(row));
    }

    milvus::UpsertResponse resp_upsert;
    status = client->Upsert(
        milvus::UpsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(upsert_rows)), resp_upsert);
    util::CheckStatus("upsert", status);

    // if the primary key is auto-id, upsert() will delete the old id and create a new id,
    // this behavior is a technical trade-off of milvus
    const auto new_ids = resp_upsert.Results().IdArray().IntIDArray();
    int64_t new_id_1 = new_ids[0];
    int64_t new_id_2 = new_ids[1];
    std::cout << "After upsert, the id " << update_id_1 << " has been updated to " << new_id_1 << std::endl;
    std::cout << "After upsert, the id " << update_id_2 << " has been updated to " << new_id_2 << std::endl;

    // query the updated items
    std::string expr = field_id + " in [" + std::to_string(new_id_1) + "," + std::to_string(new_id_2) + "]";
    auto request = milvus::QueryRequest()
                       .WithCollectionName(collection_name)
                       .WithFilter(expr)
                       .AddOutputField(field_id)
                       .AddOutputField(field_text)
                       .AddOutputField(field_vector)
                       // the SESSION level ensures that the previous dml change of this process must be
                       // visible to the next query/search of the same process.
                       .WithConsistencyLevel(milvus::ConsistencyLevel::SESSION);

    std::cout << "Query with expression: " << expr << std::endl;
    milvus::QueryResponse response;
    status = client->Query(request, response);
    util::CheckStatus("query", status);
    std::cout << "Query results:" << std::endl;
    const auto& query_results = response.Results();
    for (auto i = 0; i < query_results.GetRowCount(); i++) {
        milvus::EntityRow output_row;
        status = query_results.OutputRow(i, output_row);
        util::CheckStatus("get output row", status);
        std::cout << "\t" << output_row << std::endl;
    }

    // delete the two items
    std::cout << "Delete with expression: " << expr << std::endl;
    milvus::DeleteResponse resp_delete;
    status = client->Delete(milvus::DeleteRequest().WithCollectionName(collection_name).WithFilter(expr), resp_delete);
    util::CheckStatus("delete", status);

    // query immediatelly again with STRONG level, result must be empty.
    // set to STRONG level so that the query is executed after the inserted data is consumed by server
    request.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    status = client->Query(request, response);
    util::CheckStatus("query again with the same expression", status);
    std::cout << "Query result count: " << std::to_string(response.Results().GetRowCount()) << std::endl;

    // get the numer of rows after delete, must be 100 - 2 = 98
    // no data changed after the last query, we can use EVENTUALLY level to ignore
    // dml consistency check(in the server side)
    {
        auto request = milvus::QueryRequest()
                           .WithCollectionName(collection_name)
                           .AddOutputField("count(*)")
                           .WithConsistencyLevel(milvus::ConsistencyLevel::EVENTUALLY);

        milvus::QueryResponse response;
        status = client->Query(request, response);
        util::CheckStatus("query count(*)", status);
        std::cout << "count(*) = " << response.Results().GetRowCount() << std::endl;
    }

    client->Disconnect();
    return 0;
}
