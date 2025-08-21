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
    util::CheckStatus("Failed to connect milvus server:", status);
    std::cout << "Connect to milvus server." << std::endl;

    // drop the collection if it exists
    const std::string collection_name = "TEST_CPP_HYBRID";
    status = client->DropCollection(collection_name);

    // create a collection
    const std::string field_id = "id";
    const std::string field_text = "text";
    const std::string field_dense = "dense";
    const std::string field_sparse = "sparse";
    const uint32_t dimension = 128;

    // collection schema, create collection
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField({field_id, milvus::DataType::INT64, "id", true, false});
    collection_schema.AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR, "text").WithMaxLength(1024));
    collection_schema.AddField(
        milvus::FieldSchema(field_dense, milvus::DataType::FLOAT_VECTOR, "dense vector").WithDimension(dimension));
    collection_schema.AddField(
        milvus::FieldSchema(field_sparse, milvus::DataType::SPARSE_FLOAT_VECTOR, "sparse vector"));

    status = client->CreateCollection(collection_schema);
    util::CheckStatus("Failed to create collection:", status);
    std::cout << "Successfully create collection " << collection_name << std::endl;

    // create index
    milvus::IndexDesc index_dense(field_dense, "", milvus::IndexType::DISKANN, milvus::MetricType::L2);
    status = client->CreateIndex(collection_name, index_dense);
    util::CheckStatus("Failed to create index on dense vector field:", status);
    std::cout << "Successfully create index on dense vector field." << std::endl;

    milvus::IndexDesc index_sparse(field_sparse, "", milvus::IndexType::SPARSE_INVERTED_INDEX, milvus::MetricType::IP);
    status = client->CreateIndex(collection_name, index_sparse);
    util::CheckStatus("Failed to create index on sparse vector field:", status);
    std::cout << "Successfully create index on sparse vector field." << std::endl;

    // tell server prepare to load collection
    status = client->LoadCollection(collection_name);
    util::CheckStatus("Failed to load collection:", status);

    // insert some rows
    const int64_t row_count = 1000;
    std::vector<nlohmann::json> rows;
    for (auto i = 0; i < row_count; ++i) {
        nlohmann::json row;
        row[field_id] = i;
        row[field_text] = "text_" + std::to_string(i);
        row[field_dense] = util::GenerateFloatVector(dimension);
        row[field_sparse] = util::GenerateSparseVectorInJson(50, false);
        rows.emplace_back(std::move(row));
    }

    milvus::DmlResults dml_results;
    status = client->Insert(collection_name, "", rows, dml_results);
    util::CheckStatus("Failed to insert:", status);
    std::cout << "Successfully insert " << dml_results.InsertCount() << " rows." << std::endl;

    {
        // verify the row count of the partition is 999 by query(count(*))
        // set to STRONG level to ensure the delete request is done by server
        milvus::QueryArguments q_count{};
        q_count.SetCollectionName(collection_name);
        q_count.AddOutputField("count(*)");
        q_count.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        milvus::QueryResults count_resutl{};
        status = client->Query(q_count, count_resutl);
        util::CheckStatus("Failed to query count(*):", status);
        std::cout << "Successfully query count(*)." << std::endl;
        std::cout << "count(*) = " << count_resutl.GetRowCount() << std::endl;
    }

    {
        // do hybrid search
        milvus::HybridSearchArguments s_arguments{};
        s_arguments.SetCollectionName(collection_name);
        s_arguments.SetLimit(10);
        s_arguments.AddOutputField(field_text);
        s_arguments.AddOutputField(field_sparse);
        // set to BOUNDED level to accept data inconsistence within a time window(default is 5 seconds)
        s_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

        // sub search request 1 for dense vector
        auto sub_req1 = std::make_shared<milvus::SubSearchRequest>();
        sub_req1->SetLimit(5);
        sub_req1->SetFilter(field_id + " > 50");
        status = sub_req1->AddFloatVector(field_dense, util::GenerateFloatVector(dimension));
        util::CheckStatus("Failed to add vector to SubSearchRequest:", status);
        s_arguments.AddSubRequest(sub_req1);

        // sub search request 2 for sparse vector
        auto sub_req2 = std::make_shared<milvus::SubSearchRequest>();
        sub_req2->SetLimit(15);
        sub_req2->SetFilter(field_id + " < 100");
        status = sub_req2->AddSparseVector(field_sparse, util::GenerateSparseVector(50));
        util::CheckStatus("Failed to add vector to SubSearchRequest:", status);
        s_arguments.AddSubRequest(sub_req2);

        // define reranker
        auto reranker = std::make_shared<milvus::WeightedRerank>(std::vector<float>{0.2, 0.8});
        s_arguments.SetRerank(reranker);

        milvus::SearchResults search_results{};
        status = client->HybridSearch(s_arguments, search_results);
        util::CheckStatus("Failed to search:", status);
        std::cout << "Successfully search." << std::endl;

        for (auto& result : search_results.Results()) {
            std::cout << "Result of one target vector:" << std::endl;
            std::vector<nlohmann::json> output_rows;
            status = result.OutputRows(output_rows);
            util::CheckStatus("Failed to get output rows:", status);
            for (const auto& row : output_rows) {
                std::cout << "\t" << row << std::endl;
            }
        }
    }

    // drop collection
    status = client->DropCollection(collection_name);
    util::CheckStatus("Failed to drop collection:", status);
    std::cout << "Drop collection " << collection_name << std::endl;

    client->Disconnect();
    return 0;
}
