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

#include "milvus/MilvusClient.h"
#include "milvus/types/CollectionSchema.h"
#include "util/ExampleUtils.h"

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClient::Create();

    milvus::ConnectParam connect_param{"localhost", 19530, "root", "Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("Failed to connect milvus server:", status);
    std::cout << "Connect to milvus server." << std::endl;

    // drop the collection if it exists
    const std::string collection_name = "TEST_CPP_DML";
    status = client->DropCollection(collection_name);

    // create a collection
    const std::string field_id = "pk";
    const std::string field_vector = "vector";
    const std::string field_text = "text";
    const uint32_t dimension = 4;

    // collection schema, create collection
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField({field_id, milvus::DataType::INT64, "id", true, true});
    collection_schema.AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
    collection_schema.AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR).WithMaxLength(100));

    status = client->CreateCollection(collection_schema);
    util::CheckStatus("Failed to create collection:", status);
    std::cout << "Successfully create collection " << collection_name << std::endl;

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::AUTOINDEX, milvus::MetricType::L2);
    status = client->CreateIndex(collection_name, index_vector);
    util::CheckStatus("Failed to create index on vector field:", status);
    milvus::IndexDesc index_text(field_text, "", milvus::IndexType::INVERTED);
    status = client->CreateIndex(collection_name, index_text);
    util::CheckStatus("Failed to create index on vector field:", status);
    std::cout << "Successfully create index." << std::endl;

    // load collection
    status = client->LoadCollection(collection_name);
    util::CheckStatus("Failed to load collection:", status);
    std::cout << "Successfully load collection." << std::endl;

    // insert some rows
    const int64_t row_count = 100;
    std::vector<std::vector<float>> insert_vectors = util::GenerateFloatVectors(dimension, row_count);
    std::vector<std::string> insert_texts(row_count);
    for (auto i = 0; i < row_count; ++i) {
        insert_texts[i] = "hello world " + std::to_string(i);
    }

    std::vector<milvus::FieldDataPtr> fields_data{
        std::make_shared<milvus::VarCharFieldData>(field_text, insert_texts),
        std::make_shared<milvus::FloatVecFieldData>(field_vector, insert_vectors),
    };
    milvus::DmlResults dml_results;
    status = client->Insert(collection_name, "", fields_data, dml_results);
    util::CheckStatus("Failed to insert:", status);
    std::cout << "Successfully insert " << dml_results.IdArray().IntIDArray().size() << " rows." << std::endl;
    const auto& ids = dml_results.IdArray().IntIDArray();

    // upsert one row
    int64_t update_id_1 = ids[1];
    int64_t update_id_2 = ids[ids.size() - 1];
    std::vector<int64_t> update_ids = {update_id_1, update_id_2};
    std::vector<std::string> update_texts = {
        "this row is updated from " + std::to_string(update_id_1),
        "this row is updated from " + std::to_string(update_id_2),
    };

    std::vector<float> dummy_vector(dimension);
    for (auto d = 0; d < dimension; ++d) {
        dummy_vector[d] = 0.88;
    }
    std::vector<std::vector<float>> update_vectors = {dummy_vector, dummy_vector};
    std::vector<milvus::FieldDataPtr> update_data{
        std::make_shared<milvus::Int64FieldData>(field_id, update_ids),
        std::make_shared<milvus::VarCharFieldData>(field_text, update_texts),
        std::make_shared<milvus::FloatVecFieldData>(field_vector, update_vectors),
    };
    milvus::DmlResults update_results;
    status = client->Upsert(collection_name, "", update_data, update_results);
    util::CheckStatus("Failed to upsert:", status);
    std::cout << "Successfully upsert." << std::endl;
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
    milvus::QueryResults query_resutls{};
    status = client->Query(q_arguments, query_resutls);
    util::CheckStatus("Failed to query:", status);
    std::cout << "Successfully query." << std::endl;

    auto id_field_data = query_resutls.OutputField<milvus::Int64FieldData>(field_id);
    auto text_field_data = query_resutls.OutputField<milvus::VarCharFieldData>(field_text);
    auto vetor_field_data = query_resutls.OutputField<milvus::FloatVecFieldData>(field_vector);

    for (size_t i = 0; i < id_field_data->Count(); ++i) {
        std::cout << field_id << ":" << id_field_data->Value(i) << "\t" << field_text << ":"
                  << text_field_data->Value(i) << "\t" << field_vector << ":";
        util::PrintList(vetor_field_data->Value(i));
        std::cout << std::endl;
    }

    // delete the two items
    std::cout << "Delete with expression: " << expr << std::endl;
    milvus::DmlResults delete_results;
    status = client->Delete(collection_name, "", expr, delete_results);
    util::CheckStatus("Failed to delete:", status);
    std::cout << "Successfully delete." << std::endl;

    // query immediatelly again with STRONG level, result must be empty.
    // set to STRONG level so that the query is executed after the inserted data is consumed by server
    q_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    status = client->Query(q_arguments, query_resutls);
    util::CheckStatus("Failed to query:", status);
    std::cout << "Successfully query again with the same expression." << std::endl;

    id_field_data = query_resutls.OutputField<milvus::Int64FieldData>(field_id);
    std::cout << "Query result count: " << std::to_string(id_field_data->Count()) << std::endl;

    // get the numer of rows after delete, must be 100 - 2 = 98
    // no data changed after the last query, we can use EVENTUALLY level to ignore
    // dml consistency check(in the server side)
    milvus::QueryArguments q_count{};
    q_count.SetCollectionName(collection_name);
    q_count.AddOutputField("count(*)");
    q_count.SetConsistencyLevel(milvus::ConsistencyLevel::EVENTUALLY);

    milvus::QueryResults count_resutl{};
    status = client->Query(q_count, count_resutl);
    util::CheckStatus("Failed to query count(*):", status);
    std::cout << "Successfully query count(*)." << std::endl;
    std::cout << "count(*) = " << count_resutl.GetCountNumber() << std::endl;

    client->Disconnect();
    return 0;
}
