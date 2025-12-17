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

    const std::string collection_name = "CPP_V1_HYBRID_SEARCH";
    const std::string field_id = "id";
    const std::string field_flag = "flag";
    const std::string field_text = "text";
    const std::string field_dense = "dense";
    const std::string field_sparse = "sparse";
    const uint32_t dimension = 128;

    // collection schema, drop and create collection
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField({field_id, milvus::DataType::INT64, "id", true, false});
    collection_schema.AddField({field_flag, milvus::DataType::INT16, "flag"});
    collection_schema.AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR, "text").WithMaxLength(1024));
    collection_schema.AddField(
        milvus::FieldSchema(field_dense, milvus::DataType::FLOAT_VECTOR, "dense vector").WithDimension(dimension));
    collection_schema.AddField(
        milvus::FieldSchema(field_sparse, milvus::DataType::SPARSE_FLOAT_VECTOR, "sparse vector"));

    status = client->DropCollection(collection_name);
    status = client->CreateCollection(collection_schema);
    util::CheckStatus("create collection: " + collection_name, status);

    // create index
    milvus::IndexDesc index_dense(field_dense, "", milvus::IndexType::DISKANN, milvus::MetricType::COSINE);
    status = client->CreateIndex(collection_name, index_dense);
    util::CheckStatus("create index on dense vector field", status);

    milvus::IndexDesc index_sparse(field_sparse, "", milvus::IndexType::SPARSE_INVERTED_INDEX, milvus::MetricType::IP);
    status = client->CreateIndex(collection_name, index_sparse);
    util::CheckStatus("create index on sparse vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(collection_name);
    util::CheckStatus("load collection: " + collection_name, status);

    // insert some rows
    const int64_t row_count = 1000;
    milvus::EntityRows rows;
    for (auto i = 0; i < row_count; ++i) {
        milvus::EntityRow row;
        row[field_id] = i;
        row[field_flag] = i % 8 + 1;
        row[field_text] = "text_" + std::to_string(i);
        row[field_dense] = util::GenerateFloatVector(dimension);
        row[field_sparse] = util::GenerateSparseVectorInJson(50, false);
        rows.emplace_back(std::move(row));
    }

    milvus::DmlResults dml_results;
    status = client->Insert(collection_name, "", rows, dml_results);
    util::CheckStatus("insert", status);
    std::cout << dml_results.InsertCount() << " rows inserted" << std::endl;

    {
        // verify the row count of the partition is 999 by query(count(*))
        // set to STRONG level to ensure the delete request is done by server
        milvus::QueryArguments q_count{};
        q_count.SetCollectionName(collection_name);
        q_count.AddOutputField("count(*)");
        q_count.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        milvus::QueryResults count_result{};
        status = client->Query(q_count, count_result);
        util::CheckStatus("query count(*)", status);
        std::cout << "count(*) = " << count_result.GetRowCount() << std::endl;
    }

    {
        // do hybrid search
        milvus::HybridSearchArguments s_arguments{};
        s_arguments.SetCollectionName(collection_name);
        s_arguments.SetLimit(10);
        s_arguments.AddOutputField(field_flag);
        s_arguments.AddOutputField(field_text);
        // s_arguments.AddOutputField(field_sparse);
        // set to BOUNDED level to accept data inconsistence within a time window(default is 5 seconds)
        s_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

        // sub search request 1 for dense vector
        auto sub_req1 = std::make_shared<milvus::SubSearchRequest>();
        sub_req1->SetLimit(5);
        sub_req1->SetFilter(field_flag + " == 5");
        sub_req1->SetAnnsField(field_dense);
        sub_req1->AddFloatVector(util::GenerateFloatVector(dimension));
        s_arguments.AddSubRequest(sub_req1);

        // sub search request 2 for sparse vector
        auto sub_req2 = std::make_shared<milvus::SubSearchRequest>();
        sub_req2->SetLimit(15);
        sub_req2->SetFilter(field_flag + " in [1, 3]");
        sub_req1->SetAnnsField(field_sparse);
        sub_req2->AddSparseVector(util::GenerateSparseVector(50));
        s_arguments.AddSubRequest(sub_req2);

        // define reranker
        auto reranker = std::make_shared<milvus::WeightedRerank>(std::vector<float>{0.5, 0.5});
        s_arguments.SetRerank(reranker);

        milvus::SearchResults search_results{};
        status = client->HybridSearch(s_arguments, search_results);
        util::CheckStatus("search", status);

        for (auto& result : search_results.Results()) {
            std::cout << "Result of one target vector:" << std::endl;
            milvus::EntityRows output_rows;
            status = result.OutputRows(output_rows);
            util::CheckStatus("get output rows", status);
            for (const auto& row : output_rows) {
                std::cout << "\t" << row << std::endl;
            }
        }
    }

    // drop collection
    status = client->DropCollection(collection_name);
    util::CheckStatus("drop collection: " + collection_name, status);

    client->Disconnect();
    return 0;
}
