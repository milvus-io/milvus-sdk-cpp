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
const std::string collection_name = "TEST_CPP_GROUP_BY";
const std::string field_id = "id";
const std::string field_vector = "vector";
const std::string field_chunk = "chunk";
const std::string field_doc_id = "docId";
const uint32_t dimension = 5;

void
searchGroupBy(milvus::MilvusClientPtr& client, const std::string& group_field, int64_t limit, uint64_t group_size,
              bool strict_group_size) {
    // search with grup by docId
    std::vector<float> target_vector = {0.145292, 0.914725, 0.796505, 0.700925, 0.560520};
    milvus::SearchArguments s_arguments{};
    s_arguments.SetCollectionName(collection_name);
    s_arguments.AddFloatVector(field_vector, target_vector);
    s_arguments.SetLimit(limit);
    s_arguments.AddOutputField(field_doc_id);
    // session level ensure that the data inserted by this client is visible
    s_arguments.SetConsistencyLevel(milvus::ConsistencyLevel::SESSION);

    if (!group_field.empty()) {
        s_arguments.SetGroupByField(group_field);
        if (group_size > 0) {
            s_arguments.SetGroupSize(group_size);
            s_arguments.SetStrictGroupSize(strict_group_size);
        }
    }

    std::cout << "\n===================================================================================" << std::endl;
    std::cout << "Search with group by field: " << (group_field.empty() ? "null" : group_field)
              << ", group size: " << group_size << ", strict: " << strict_group_size << ", limit: " << limit
              << std::endl;
    milvus::SearchResults search_results{};
    auto status = client->Search(s_arguments, search_results);
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

}  // namespace

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClient::Create();

    milvus::ConnectParam connect_param{"localhost", 19530, "root", "Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    // collection schema, drop and create collection
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.AddField({field_id, milvus::DataType::INT64, "", true, false});
    collection_schema.AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
    collection_schema.AddField(milvus::FieldSchema(field_chunk, milvus::DataType::VARCHAR).WithMaxLength(128));
    collection_schema.AddField(milvus::FieldSchema(field_doc_id, milvus::DataType::INT32));

    status = client->DropCollection(collection_name);
    status = client->CreateCollection(collection_schema);
    util::CheckStatus("create collection: " + collection_name, status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::FLAT, milvus::MetricType::COSINE);
    status = client->CreateIndex(collection_name, index_vector);
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(collection_name);
    util::CheckStatus("load collection: " + collection_name, status);

    {
        // insert data row by row
        std::vector<std::string> rows = {
            R"({"id": 0, "vector": [0.3580376395471989, -0.6023495712049978, 0.18414012509913835, -0.26286205330961354, 0.9029438446296592], "chunk": "pink_8682", "docId": 1})",
            R"({"id": 1, "vector": [0.19886812562848388, 0.06023560599112088, 0.6976963061752597, 0.2614474506242501, 0.838729485096104], "chunk": "red_7025", "docId": 5})",
            R"({"id": 2, "vector": [0.43742130801983836, -0.5597502546264526, 0.6457887650909682, 0.7894058910881185, 0.20785793220625592], "chunk": "orange_6781", "docId": 2})",
            R"({"id": 3, "vector": [0.3172005263489739, 0.9719044792798428, -0.36981146090600725, -0.4860894583077995, 0.95791889146345], "chunk": "pink_9298", "docId": 3})",
            R"({"id": 4, "vector": [0.4452349528804562, -0.8757026943054742, 0.8220779437047674, 0.46406290649483184, 0.30337481143159106], "chunk": "red_4794", "docId": 3})",
            R"({"id": 5, "vector": [0.985825131989184, -0.8144651566660419, 0.6299267002202009, 0.1206906911183383, -0.1446277761879955], "chunk": "yellow_4222", "docId": 4})",
            R"({"id": 6, "vector": [0.8371977790571115, -0.015764369584852833, -0.31062937026679327, -0.562666951622192, -0.8984947637863987], "chunk": "red_9392", "docId": 1})",
            R"({"id": 7, "vector": [-0.33445148015177995, -0.2567135004164067, 0.8987539745369246, 0.9402995886420709, 0.5378064918413052], "chunk": "grey_8510", "docId": 2})",
            R"({"id": 8, "vector": [0.39524717779832685, 0.4000257286739164, -0.5890507376891594, -0.8650502298996872, -0.6140360785406336], "chunk": "white_9381", "docId": 5})",
            R"({"id": 9, "vector": [0.5718280481994695, 0.24070317428066512, -0.3737913482606834, -0.06726932177492717, -0.6980531615588608], "chunk": "purple_4976", "docId": 3})",
        };
        for (auto& row : rows) {
            milvus::EntityRow entity = milvus::EntityRow::parse(row);
            milvus::DmlResults dml_results;
            status = client->Insert(collection_name, "", milvus::EntityRows{entity}, dml_results);
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
