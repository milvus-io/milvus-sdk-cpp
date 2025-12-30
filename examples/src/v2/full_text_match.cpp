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

namespace {
const char* const collection_name = "CPP_V2_FULL_TEXT_SEARCH";
const char* const field_id = "id";
const char* const field_vector = "vector";
const char* const field_text = "text";

void
buildCollection(milvus::MilvusClientV2Ptr& client) {
    // collection schema, drop and create collection
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->AddField({field_id, milvus::DataType::INT64, "", true, false});
    collection_schema->AddField(milvus::FieldSchema(field_vector, milvus::DataType::SPARSE_FLOAT_VECTOR));
    collection_schema->AddField(
        milvus::FieldSchema(field_text, milvus::DataType::VARCHAR).WithMaxLength(65535).EnableAnalyzer(true));

    // define the BM25 function, milvus will automatically generate sparse vector by BM25 algorithm for the "text" field
    // the sparse vectors are stored in the "vector" field, and are invisible to users
    milvus::FunctionPtr function = std::make_shared<milvus::Function>("function_bm25", milvus::FunctionType::BM25);
    function->AddInputFieldName(field_text);
    function->AddOutputFieldName(field_vector);
    collection_schema->AddFunction(function);

    auto status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(collection_schema));
    util::CheckStatus(std::string("create collection: ") + collection_name, status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::SPARSE_INVERTED_INDEX,
                                   milvus::MetricType::BM25);
    status = client->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_vector)));
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
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
        rows.emplace_back(std::move(row));
    }

    milvus::InsertResponse resp_insert;
    status = client->Insert(milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)),
                            resp_insert);
    util::CheckStatus("insert", status);

    // get row count
    auto request = milvus::QueryRequest()
                       .WithCollectionName(collection_name)
                       .AddOutputField("count(*)")
                       .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::QueryResponse response;
    status = client->Query(request, response);
    util::CheckStatus("query count(*)", status);
    std::cout << "count(*) = " << response.Results().GetRowCount() << std::endl;
}

void
searchByText(milvus::MilvusClientV2Ptr& client, std::string text) {
    std::cout << "================================================================" << std::endl;
    std::cout << "Search by text: " << text << std::endl;

    auto request = milvus::SearchRequest()
                       .WithCollectionName(collection_name)
                       .AddEmbeddedText(text)
                       .WithLimit(5)
                       .WithAnnsField(field_vector)
                       .AddOutputField(field_id)
                       .AddOutputField(field_text)
                       .WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::SearchResponse response;
    auto status = client->Search(request, response);
    util::CheckStatus("search", status);

    auto& result = response.Results().Results().at(0);
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

    auto client = milvus::MilvusClientV2::Create();

    milvus::ConnectParam connect_param{"localhost", 19530, "root", "Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    buildCollection(client);

    searchByText(client, "moon and earth distance");
    searchByText(client, "Milvus vector database");

    client->Disconnect();
    return 0;
}
