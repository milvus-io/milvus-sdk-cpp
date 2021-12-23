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

#include "MilvusClient.h"
#include "types/CollectionSchema.h"

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClient::Create();

    milvus::ConnectParam connect_param{"localhost", 19530};
    auto status = client->Connect(connect_param);
    if (!status.IsOk()) {
        std::cout << "Failed to connect milvus server: " << status.Message() << std::endl;
        return 0;
    }

    std::cout << "Connect to milvus server." << std::endl;

    milvus::CollectionSchema collection_schema("TEST");

    milvus::FieldSchema field_schema_1("identity", milvus::DataType::INT64, "user id", true, true);
    milvus::FieldSchema field_schema_2("age", milvus::DataType::INT8, "user age");
    collection_schema.AddField(field_schema_1);
    collection_schema.AddField(field_schema_2);

    status = client->CreateCollection(collection_schema);
    if (!status.IsOk()) {
        std::cout << "Failed to create collection: " << status.Message() << std::endl;
    }

    std::cout << "Successfully create collection." << std::endl;

    printf("Example stop...\n");
    return 0;
}
