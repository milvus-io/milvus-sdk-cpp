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

    const std::string collection_name = "CPP_V2_STRUCT";
    const std::string field_id = "id";
    const std::string field_vector = "vector";
    const std::string field_struct = "struct_field";
    const std::string field_struct_int32 = "struct_int32";
    const std::string field_struct_varchar = "struct_varchar";
    const std::string field_struct_vector = "struct_vector";

    const uint32_t dimension = 4;
    const int64_t struct_capacity = 10;

    // collection schema, drop and create collection
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>(collection_name);
    collection_schema->AddField(milvus::FieldSchema(field_id, milvus::DataType::INT64, "id", true, false));
    collection_schema->AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR, "face signature").WithDimension(dimension));

    milvus::StructFieldSchema struct_schema =
        milvus::StructFieldSchema()
            .WithName(field_struct)
            .WithMaxCapacity(struct_capacity)
            .AddField(milvus::FieldSchema(field_struct_int32, milvus::DataType::INT32))
            .AddField(milvus::FieldSchema(field_struct_varchar, milvus::DataType::VARCHAR).WithMaxLength(512))
            .AddField(
                milvus::FieldSchema(field_struct_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
    collection_schema->AddStructField(std::move(struct_schema));

    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(milvus::CreateCollectionRequest().WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + collection_name, status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::FLAT, milvus::MetricType::COSINE);
    // for struct vector field, the name format is: struct_field[struct_vector_field]
    std::string st_vector_name = field_struct + "[" + field_struct_vector + "]";
    milvus::IndexDesc index_struct(st_vector_name, "", milvus::IndexType::HNSW, milvus::MetricType::MAX_SIM_COSINE);
    status = client->CreateIndex(milvus::CreateIndexRequest()
                                     .WithCollectionName(collection_name)
                                     .AddIndex(std::move(index_vector))
                                     .AddIndex(std::move(index_struct)));
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection: " + collection_name, status);

    // insert some rows by row-based
    const int64_t row_count = 1;
    milvus::EntityRows rows;
    for (auto i = 0; i < row_count; ++i) {
        milvus::EntityRow row;
        row[field_id] = i;
        row[field_vector] = util::GenerateFloatVector(dimension);

        std::vector<milvus::EntityRow> struct_list;
        for (auto k = 0; k <= i; k++) {
            milvus::EntityRow st;
            st[field_struct_int32] = k;
            st[field_struct_varchar] = "str_" + std::to_string(k);
            st[field_struct_vector] = util::GenerateFloatVector(dimension);
            struct_list.emplace_back(std::move(st));
        }
        row[field_struct] = struct_list;

        rows.emplace_back(std::move(row));
    }

    milvus::InsertResponse resp_insert;
    status = client->Insert(milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)),
                            resp_insert);
    util::CheckStatus("insert", status);
    std::cout << resp_insert.Results().IdArray().StrIDArray().size() << " rows inserted." << std::endl;

    {
        // query some items wihtout filtering
        milvus::QueryRequest request;
        request.SetCollectionName(collection_name);
        request.AddOutputField(field_id);
        request.AddOutputField(field_struct);
        request.SetLimit(5);
        // set to strong level so that the query is executed after the inserted data is consumed by server
        request.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        milvus::QueryResponse response;
        status = client->Query(request, response);
        util::CheckStatus("query", status);

        milvus::EntityRows output_rows;
        status = response.Results().OutputRows(output_rows);
        util::CheckStatus("get output rows", status);
        std::cout << "Query results:" << std::endl;
        for (const auto& row : output_rows) {
            std::cout << "\t" << row << std::endl;
        }
    }

    // {
    //     // do search
    //     milvus::SearchRequest request;
    //     request.SetCollectionName(collection_name);
    //     request.SetLimit(3);
    //     request.AddOutputField(field_id);
    //     request.AddOutputField(field_struct);

    //     auto q_number_1 = util::RandomeValue<int64_t>(0, row_count - 1);
    //     auto q_number_2 = util::RandomeValue<int64_t>(0, row_count - 1);
    //     request.AddFloatVector(field_vector, rows[q_number_1][field_vector]);
    //     request.AddFloatVector(field_vector, rows[q_number_2][field_vector]);
    //     std::cout << "Searching the No." << q_number_1 << " and No." << q_number_2 << std::endl;

    //     milvus::SearchResponse response;
    //     status = client->Search(request, response);
    //     util::CheckStatus("search", status);

    //     for (auto& result : response.Results().Results()) {
    //         std::cout << "Result of one target vector:" << std::endl;
    //         milvus::EntityRows output_rows;
    //         status = result.OutputRows(output_rows);
    //         util::CheckStatus("get output rows", status);
    //         std::cout << "Result of one target vector:" << std::endl;
    //         for (const auto& row : output_rows) {
    //             std::cout << "\t" << row << std::endl;
    //         }
    //     }
    // }

    client->Disconnect();
    return 0;
}
