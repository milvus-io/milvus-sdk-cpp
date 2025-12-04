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
#include "milvus/MilvusClient.h"

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClient::Create();

    milvus::ConnectParam connect_param{"localhost", 19530, "root", "Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    const std::string collection_name = "CPP_V1_DML";
    const std::string field_id = "pk";
    const std::string field_vector = "vector";
    const std::string field_text = "text";
    const uint32_t dimension = 4;

    // collection schema, drop and create collection
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField({field_id, milvus::DataType::INT64, "id", true, true});
    collection_schema.AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
    collection_schema.AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR).WithMaxLength(100));

    status = client->DropCollection(collection_name);
    status = client->CreateCollection(collection_schema);
    util::CheckStatus("create collection: " + collection_name, status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::HNSW, milvus::MetricType::L2);
    index_vector.AddExtraParam("M", "64");
    index_vector.AddExtraParam("efConstruction", "200");
    status = client->CreateIndex(collection_name, index_vector);
    util::CheckStatus("create index on vector field", status);
    milvus::IndexDesc index_text(field_text, "", milvus::IndexType::INVERTED);
    status = client->CreateIndex(collection_name, index_text);
    util::CheckStatus("create index on varchar field", status);

    // load collection
    status = client->LoadCollection(collection_name);
    util::CheckStatus("load collection: " + collection_name, status);

    {
        // insert somes rows by column-based
        auto texts = std::vector<std::string>{"column-based-1", "column-based-2"};
        auto vectors = util::GenerateFloatVectors(dimension, 2);
        std::vector<milvus::FieldDataPtr> fields_data{
            std::make_shared<milvus::VarCharFieldData>(field_text, texts),
            std::make_shared<milvus::FloatVecFieldData>(field_vector, vectors)};
        milvus::DmlResults dml_results;
        status = client->Insert(collection_name, "", fields_data, dml_results);
        util::CheckStatus("insert", status);
        std::cout << dml_results.InsertCount() << " rows inserted by column-based." << std::endl;
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

    milvus::DmlResults dml_results;
    status = client->Insert(collection_name, "", rows, dml_results);
    util::CheckStatus("insert", status);
    std::cout << dml_results.InsertCount() << " rows inserted by row-based." << std::endl;
    const auto& ids = dml_results.IdArray().IntIDArray();

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

    milvus::DmlResults update_results;
    status = client->Upsert(collection_name, "", upsert_rows, update_results);
    util::CheckStatus("upsert", status);

    // if the primary key is auto-id, upsert() will delete the old id and create a new id,
    // this behavior is a technical trade-off of milvus
    const auto new_ids = update_results.IdArray().IntIDArray();
    int64_t new_id_1 = new_ids[0];
    int64_t new_id_2 = new_ids[1];
    std::cout << "After upsert, the id " << update_id_1 << " has been updated to " << new_id_1 << std::endl;
    std::cout << "After upsert, the id " << update_id_2 << " has been updated to " << new_id_2 << std::endl;

    // query the updated items
    std::string expr = field_id + " in [" + std::to_string(new_id_1) + "," + std::to_string(new_id_2) + "]";
    milvus::QueryArguments q_arguments{};
    q_arguments.SetCollectionName(collection_name);
    q_arguments.SetFilter(expr);
    q_arguments.AddOutputField(field_id);
    q_arguments.AddOutputField(field_text);
    q_arguments.AddOutputField(field_vector);
    // the SESSION level ensures that the previous dml change of this process must be
    // visible to the next query/search of the same process.
    q_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::SESSION);

    std::cout << "Query with expression: " << expr << std::endl;
    milvus::QueryResults query_results{};
    status = client->Query(q_arguments, query_results);
    util::CheckStatus("query", status);
    std::cout << "Query results:" << std::endl;
    for (auto i = 0; i < query_results.GetRowCount(); i++) {
        milvus::EntityRow output_row;
        status = query_results.OutputRow(i, output_row);
        util::CheckStatus("get output row", status);
        std::cout << "\t" << output_row << std::endl;
    }

    // delete the two items
    std::cout << "Delete with expression: " << expr << std::endl;
    milvus::DmlResults delete_results;
    status = client->Delete(collection_name, "", expr, delete_results);
    util::CheckStatus("delete", status);

    // query immediatelly again with STRONG level, result must be empty.
    // set to STRONG level so that the query is executed after the inserted data is consumed by server
    q_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    status = client->Query(q_arguments, query_results);
    util::CheckStatus("query again with the same expression", status);
    std::cout << "Query result count: " << std::to_string(query_results.GetRowCount()) << std::endl;

    // get the numer of rows after delete, must be 100 - 2 = 98
    // no data changed after the last query, we can use EVENTUALLY level to ignore
    // dml consistency check(in the server side)
    milvus::QueryArguments q_count{};
    q_count.SetCollectionName(collection_name);
    q_count.AddOutputField("count(*)");
    q_count.SetConsistencyLevel(milvus::ConsistencyLevel::EVENTUALLY);

    milvus::QueryResults count_result{};
    status = client->Query(q_count, count_result);
    util::CheckStatus("query count(*)", status);
    std::cout << "count(*) = " << count_result.GetRowCount() << std::endl;

    client->Disconnect();
    return 0;
}
