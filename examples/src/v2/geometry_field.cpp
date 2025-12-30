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

const char* const collection_name = "CPP_V2_GEOMETRY_FIELD";
const char* const field_id = "id";
const char* const field_vector = "vector";
const char* const field_geo = "geo";
const uint32_t dimension = 4;

void
insertGeometry(milvus::MilvusClientV2Ptr& client, std::string geometry) {
    milvus::EntityRow row;
    row[field_vector] = util::GenerateFloatVector(dimension);
    row[field_geo] = std::move(geometry);

    milvus::InsertResponse resp_insert;
    auto status = client->Insert(milvus::InsertRequest().WithCollectionName(collection_name).AddRowData(std::move(row)),
                                 resp_insert);
    util::CheckStatus("insert", status);
    std::cout << resp_insert.Results().InsertCount() << " rows inserted by row-based." << std::endl;
}

void
query(milvus::MilvusClientV2Ptr& client, std::string filter) {
    auto request = milvus::QueryRequest().WithCollectionName(collection_name).AddOutputField("*").WithFilter(filter);

    std::cout << "\n========= Query with filter: " << request.Filter() << std::endl;
    milvus::QueryResponse response;
    auto status = client->Query(request, response);
    util::CheckStatus("query", status);

    milvus::EntityRows output_rows;
    status = response.Results().OutputRows(output_rows);
    util::CheckStatus("get output rows", status);
    std::cout << "Query results:" << std::endl;
    for (const auto& row : output_rows) {
        std::cout << "\t" << row << std::endl;
    }
}

void
search(milvus::MilvusClientV2Ptr& client, std::string filter) {
    auto request = milvus::SearchRequest()
                       .WithCollectionName(collection_name)
                       .WithFilter(filter)
                       .WithLimit(20)
                       .AddOutputField(field_geo)
                       .AddFloatVector(util::GenerateFloatVector(dimension));

    std::cout << "\n========= Search with filter: " << request.Filter() << std::endl;
    milvus::SearchResponse response;
    auto status = client->Search(request, response);
    util::CheckStatus("search", status);

    for (auto& result : response.Results().Results()) {
        std::cout << "Result of one target vector:" << std::endl;
        milvus::EntityRows output_rows;
        status = result.OutputRows(output_rows);
        util::CheckStatus("get output rows", status);
        for (const auto& row : output_rows) {
            std::cout << "\t" << row << std::endl;
        }
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

    // collection schema, drop and create collection
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->SetEnableDynamicField(true);
    collection_schema->AddField({field_id, milvus::DataType::INT64, "", true, true});
    collection_schema->AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
    collection_schema->AddField(milvus::FieldSchema(field_geo, milvus::DataType::GEOMETRY));

    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + std::string(collection_name), status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::HNSW, milvus::MetricType::L2);
    status = client->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_vector)));
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection: " + std::string(collection_name), status);

    // insert some rows
    insertGeometry(client, "POINT (1 1)");
    insertGeometry(client, "LINESTRING (10 10, 10 30, 40 40)");
    insertGeometry(client, "POLYGON ((0 100, 100 100, 100 50, 0 50, 0 100))");

    {
        // get row count
        milvus::QueryResponse response;
        status = client->Query(milvus::QueryRequest()
                                   .WithCollectionName(collection_name)
                                   .AddOutputField("count(*)")
                                   .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG),
                               response);
        util::CheckStatus("query count(*)", status);
        std::cout << "count(*) = " << response.Results().GetRowCount() << std::endl;
    }

    // search/query with geometry operators
    std::string geo_field_name = field_geo;
    query(client, "ST_EQUALS(" + geo_field_name + ", 'POINT (1 1)')");
    query(client, "ST_TOUCHES(" + geo_field_name + ", 'LINESTRING (0 50, 0 100)')");
    query(client, "ST_CONTAINS(" + geo_field_name + ", 'POINT (70 70)')");
    query(client, "ST_CROSSES(" + geo_field_name + ", 'LINESTRING (20 0, 20 100)')");
    query(client, "ST_WITHIN(" + geo_field_name + ", 'POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))')");

    search(client, "ST_EQUALS(" + geo_field_name + ", 'POINT (1 1)')");
    search(client, "ST_TOUCHES(" + geo_field_name + ", 'LINESTRING (0 50, 0 100)')");
    search(client, "ST_CONTAINS(" + geo_field_name + ", 'POINT (70 70)')");
    search(client, "ST_CROSSES(" + geo_field_name + ", 'LINESTRING (20 0, 20 100)')");
    search(client, "ST_WITHIN(" + geo_field_name + ", 'POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))')");

    client->Disconnect();
    return 0;
}
