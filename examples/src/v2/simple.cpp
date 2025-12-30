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

    const std::string collection_name = "CPP_V2_SIMPLE";
    const std::string field_id = "pk";
    const std::string field_vector = "embedding";
    const uint32_t dimension = 128;

    // create a simple collection, the collection has only two fields: primary field and vector field
    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(milvus::CreateSimpleCollectionRequest()
                                          .WithCollectionName(collection_name)
                                          .WithPrimaryFieldName(field_id)
                                          .WithVectorFieldName(field_vector)
                                          .WithDimension(dimension));
    util::CheckStatus("create simple collection: " + collection_name, status);

    // insert some rows
    const int64_t row_count = 100;
    milvus::EntityRows rows;
    for (auto i = 0; i < row_count; ++i) {
        milvus::EntityRow row;
        row[field_id] = i;
        row[field_vector] = util::GenerateFloatVector(dimension);
        rows.emplace_back(std::move(row));
    }

    milvus::InsertResponse resp_insert;
    status = client->Insert(milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)),
                            resp_insert);
    util::CheckStatus("insert", status);
    std::cout << resp_insert.Results().InsertCount() << " rows inserted by row-based." << std::endl;

    // search
    auto request = milvus::SearchRequest()
                       .WithCollectionName(collection_name)
                       .WithLimit(3)
                       .WithAnnsField(field_vector)
                       .AddFloatVector(util::GenerateFloatVector(dimension))
                       .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::SearchResponse response;
    status = client->Search(request, response);
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

    client->Disconnect();
    return 0;
}
