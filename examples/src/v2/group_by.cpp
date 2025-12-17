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
const std::string collection_name = "CPP_V2_GROUP_BY";
const std::string field_id = "id";
const std::string field_vector = "vector";
const std::string field_chunk = "chunk";
const std::string field_doc_id = "docId";
const uint32_t dimension = 5;

void
searchGroupBy(milvus::MilvusClientV2Ptr& client, const std::string& group_field, int64_t limit, int64_t group_size,
              bool strict_group_size) {
    // search with grup by docId
    std::vector<float> target_vector = {0.145292, 0.914725, 0.796505, 0.700925, 0.560520};
    auto request = milvus::SearchRequest()
                       .WithCollectionName(collection_name)
                       .WithAnnsField(field_vector)
                       .AddFloatVector(target_vector)
                       .WithLimit(limit)
                       .AddOutputField(field_doc_id)
                       // session level ensure that the data inserted by this client is visible
                       .WithConsistencyLevel(milvus::ConsistencyLevel::SESSION);

    if (!group_field.empty()) {
        request.SetGroupByField(group_field);
        if (group_size > 0) {
            request.SetGroupSize(group_size);
            request.SetStrictGroupSize(strict_group_size);
        }
    }

    std::cout << "\n===================================================================================" << std::endl;
    std::cout << "Search with group by field: " << (group_field.empty() ? "null" : group_field)
              << ", group size: " << group_size << ", strict: " << strict_group_size << ", limit: " << limit
              << std::endl;
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
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>(collection_name);
    collection_schema->AddField({field_id, milvus::DataType::INT64, "", true, false});
    collection_schema->AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
    collection_schema->AddField(milvus::FieldSchema(field_chunk, milvus::DataType::VARCHAR).WithMaxLength(128));
    collection_schema->AddField(milvus::FieldSchema(field_doc_id, milvus::DataType::INT32));

    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(milvus::CreateCollectionRequest().WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + collection_name, status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::FLAT, milvus::MetricType::COSINE);
    status = client->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_vector)));
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection: " + collection_name, status);

    {
        // insert data row by row
        std::vector<std::string> rows = {
            R"({"id": 0, "vector": [0.358037, -0.602349, 0.184140, -0.262862, 0.902943], "chunk": "pink_8682", "docId": 1})",
            R"({"id": 1, "vector": [0.198868, 0.060235, 0.697696, 0.261447, 0.838729], "chunk": "red_7025", "docId": 5})",
            R"({"id": 2, "vector": [0.437421, -0.559750, 0.645788, 0.789405, 0.207857], "chunk": "orange_6781", "docId": 2})",
            R"({"id": 3, "vector": [0.317200, 0.971904, -0.369811, 0.120690, -0.144627], "chunk": "yellow_4222", "docId": 4})",
            R"({"id": 4, "vector": [0.837197, -0.015764, -0.310629, -0.562666, -0.898494], "chunk": "red_9392", "docId": 1})",
            R"({"id": 5, "vector": [-0.33445, -0.256713, 0.898753, 0.940299, 0.537806], "chunk": "grey_8510", "docId": 2})",
            R"({"id": 6, "vector": [0.395247, 0.400025, -0.589050, -0.865050, -0.6140360], "chunk": "white_9381", "docId": 5})",
            R"({"id": 7, "vector": [0.571828, 0.240703, -0.373791, -0.067269, -0.6980531], "chunk": "purple_4976", "docId": 3})",
        };
        for (auto& row : rows) {
            milvus::EntityRow entity = milvus::EntityRow::parse(row);
            milvus::InsertResponse resp_insert;
            status = client->Insert(
                milvus::InsertRequest().WithCollectionName(collection_name).AddRowData(std::move(entity)), resp_insert);
            util::CheckStatus("insert", status);
        }
        std::cout << rows.size() << " rows inserted." << std::endl;
    }

    // search without group by, limit 3
    searchGroupBy(client, "", 3, 0, false);

    // search group by docId, limit 3, group size 1, strict false
    searchGroupBy(client, field_doc_id, 3, 1, false);

    // search group by docId, limit 3, group size 3, strict false
    searchGroupBy(client, field_doc_id, 3, 2, false);

    // search group by docId, limit 3, group size 3, strict true
    searchGroupBy(client, field_doc_id, 3, 2, true);

    // search group by docId, limit 4, group size 3, strict false
    searchGroupBy(client, field_doc_id, 4, 3, false);

    // search group by docId, limit 4, group size 3, strict false
    searchGroupBy(client, field_doc_id, 4, 3, true);

    client->Disconnect();
    return 0;
}
