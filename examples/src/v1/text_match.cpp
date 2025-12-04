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

namespace {
const char* const collection_name = "CPP_V1_TEXT_MATCH";
const char* const field_id = "id";
const char* const field_vector = "vector";
const char* const field_text = "text";
const uint32_t dimension = 128;

void
buildCollection(milvus::MilvusClientPtr& client) {
    // collection schema, drop and create collection
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField({field_id, milvus::DataType::INT64, "", true, false});
    collection_schema.AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
    collection_schema.AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR)
                                   .WithMaxLength(1024)
                                   .EnableAnalyzer(true)
                                   .EnableMatch(true));

    auto status = client->DropCollection(collection_name);
    status = client->CreateCollection(collection_schema);
    util::CheckStatus(std::string("create collection: ") + collection_name, status);

    // create index
    milvus::IndexDesc index(field_vector, "", milvus::IndexType::IVF_FLAT, milvus::MetricType::COSINE);
    status = client->CreateIndex(collection_name, index);
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(collection_name);
    util::CheckStatus(std::string("load collection: ") + collection_name, status);

    // insert some rows by row-based
    const std::vector<std::string> text_content = {
        "Milvus is an open-source vector database",
        "AI applications help people better life",
        "Will the electric car replace gas-powered car?",
        "LangChain is a composable framework to build with LLMs. Milvus is integrated into LangChain.",
        "RAG is the process of optimizing the output of a large language model",
        "Newton is one of the greatest scientist of human history",
        "Metric type L2 is Euclidean distance",
        "Embeddings represent real-world objects, like words, images, or videos, in a form that computers can process.",
        "The moon is 384,400 km distance away from earth",
        "Milvus supports L2 distance and IP similarity for float vector.",
    };

    milvus::EntityRows rows;
    for (auto i = 0; i < text_content.size(); i++) {
        milvus::EntityRow row;
        row[field_id] = i;
        row[field_text] = text_content.at(i);
        row[field_vector] = util::GenerateFloatVector(dimension);
        rows.emplace_back(std::move(row));
    }

    milvus::DmlResults dml_results;
    status = client->Insert(collection_name, "", rows, dml_results);
    util::CheckStatus("insert", status);

    // get row count
    milvus::QueryArguments q_count{};
    q_count.SetCollectionName(collection_name);
    q_count.AddOutputField("count(*)");
    q_count.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResults count_result{};
    status = client->Query(q_count, count_result);
    util::CheckStatus("query count(*)", status);
    std::cout << "count(*) = " << count_result.GetRowCount() << std::endl;
}

void
queryWithFilter(milvus::MilvusClientPtr& client, std::string filter) {
    std::cout << "================================================================" << std::endl;
    std::cout << "Query with filter: " << filter << std::endl;

    milvus::QueryArguments q_arguments{};
    q_arguments.SetCollectionName(collection_name);
    q_arguments.SetFilter(filter);
    q_arguments.AddOutputField(field_id);
    q_arguments.AddOutputField(field_text);
    q_arguments.SetLimit(50);
    q_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::QueryResults query_results{};
    auto status = client->Query(q_arguments, query_results);
    util::CheckStatus("query", status);

    milvus::EntityRows output_rows;
    status = query_results.OutputRows(output_rows);
    util::CheckStatus("get output rows", status);
    for (const auto& row : output_rows) {
        std::cout << "\t" << row << std::endl;
    }
}

void
searchWithFilter(milvus::MilvusClientPtr& client, std::string filter) {
    std::cout << "================================================================" << std::endl;
    std::cout << "Search with filter: " << filter << std::endl;

    milvus::SearchArguments s_arguments{};
    s_arguments.SetCollectionName(collection_name);
    s_arguments.SetFilter(filter);
    s_arguments.SetLimit(50);
    s_arguments.AddOutputField(field_id);
    s_arguments.AddOutputField(field_text);
    s_arguments.AddFloatVector(field_vector, util::GenerateFloatVector(dimension));
    s_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::SearchResults search_results{};
    auto status = client->Search(s_arguments, search_results);
    util::CheckStatus("search", status);

    auto& result = search_results.Results().at(0);
    milvus::EntityRows output_rows;
    status = result.OutputRows(output_rows);
    util::CheckStatus("get output rows", status);
    for (const auto& row : output_rows) {
        std::cout << "\t" << row << std::endl;
    }
}

}  // namespace

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClient::Create();

    milvus::ConnectParam connect_param{"localhost", 19530, "root", "Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    buildCollection(client);

    // TEXT_MATCH requires the data is persisted, technical limitation
    client->Flush(std::vector<std::string>{collection_name});

    // query with TEXT_MATCH
    queryWithFilter(client, R"(TEXT_MATCH(text, "distance"))");
    queryWithFilter(client, R"(TEXT_MATCH(text, "Milvus") or TEXT_MATCH(text, "distance"))");
    queryWithFilter(client, R"(TEXT_MATCH(text, "Euclidean") and TEXT_MATCH(text, "distance"))");

    // search with TEXT_MATCH
    searchWithFilter(client, R"(TEXT_MATCH(text, "distance"))");
    searchWithFilter(client, R"(TEXT_MATCH(text, "Euclidean distance"))");
    searchWithFilter(client, R"(TEXT_MATCH(text, "vector database"))");

    client->Disconnect();
    return 0;
}
