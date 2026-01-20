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

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {

const char* const collection_name = "CPP_V2_TIMESTAMP_FIELD";
const char* const field_id = "id";
const char* const field_vector = "vector";
const char* const field_timestamp = "tsz";
const uint32_t dimension = 4;

std::string
pad(int num, int width) {
    std::ostringstream oss;
    oss << std::setw(width) << std::setfill('0') << num;
    return oss.str();
}

std::string
formatDateWithTimezone(int year, int month, int day, int hour, int minute, int second,
                       std::string timezoneOffset = "+08:00") {
    std::string ts = std::to_string(year) + "-" + pad(month, 2) + "-" + pad(day, 2) + "T" + pad(hour, 2) + ":" +
                     pad(minute, 2) + ":" + pad(second, 2) + timezoneOffset;
    return ts;
}

void
insertData(milvus::MilvusClientV2Ptr& client) {
    milvus::EntityRows rows;
    std::cout << "\nInsert timezones" << std::endl;
    for (auto i = 0; i < 10; i++) {
        milvus::EntityRow row;
        row[field_id] = i;
        row[field_vector] = util::GenerateFloatVector(dimension);
        std::string ts = formatDateWithTimezone(2025, 01, i + 1, 0, 0, 0);
        row[field_timestamp] = ts;
        rows.emplace_back(std::move(row));
        std::cout << "\t" << ts << std::endl;
    }

    milvus::InsertResponse resp_insert;
    auto status = client->Insert(
        milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)), resp_insert);
    util::CheckStatus("insert", status);
    std::cout << resp_insert.Results().InsertCount() << " rows inserted by row-based." << std::endl;
}

void
query(milvus::MilvusClientV2Ptr& client, const std::string& timezone) {
    auto request = milvus::QueryRequest()
                       .WithCollectionName(collection_name)
                       .WithLimit(3)
                       .WithTimezone(timezone)
                       .AddOutputField(field_timestamp);
    milvus::QueryResponse response;
    auto status = client->Query(request, response);
    util::CheckStatus("query", status);

    milvus::EntityRows output_rows;
    status = response.Results().OutputRows(output_rows);
    util::CheckStatus("get output rows", status);
    std::cout << "\nQuery results:" << std::endl;
    for (const auto& row : output_rows) {
        std::cout << "\t" << row << std::endl;
    }
}

void
search(milvus::MilvusClientV2Ptr& client, const std::string& timezone) {
    auto request = milvus::SearchRequest()
                       .WithCollectionName(collection_name)
                       .WithLimit(3)
                       .WithTimezone(timezone)
                       .AddOutputField(field_timestamp)
                       .AddFloatVector(util::GenerateFloatVector(dimension));
    milvus::SearchResponse response;
    auto status = client->Search(request, response);
    util::CheckStatus("search", status);
    std::cout << "\nSearch results:" << std::endl;
    for (auto& result : response.Results().Results()) {
        milvus::EntityRows output_rows;
        status = result.OutputRows(output_rows);
        util::CheckStatus("get output rows", status);
        for (const auto& row : output_rows) {
            std::cout << "\t" << row << std::endl;
        }
    }
}

void
hybridSearch(milvus::MilvusClientV2Ptr& client, const std::string& timezone) {
    // this collection only has one vector field, this example demos the usage of timestamptz field and timezone search,
    // so we only have one SubSearchRequest in this hybrid search.
    auto request =
        milvus::HybridSearchRequest().WithCollectionName(collection_name).WithLimit(3).AddOutputField(field_timestamp);

    auto sub_req = milvus::SubSearchRequest()
                       .WithLimit(5)
                       .WithAnnsField(field_vector)
                       .WithTimezone(timezone)
                       .AddFloatVector(util::GenerateFloatVector(dimension));
    request.AddSubRequest(std::make_shared<milvus::SubSearchRequest>(std::move(sub_req)));

    // define reranker
    auto reranker = std::make_shared<milvus::RRFRerank>(5);
    request.SetRerank(reranker);

    milvus::SearchResponse response;
    auto status = client->HybridSearch(request, response);
    util::CheckStatus("search", status);
    std::cout << "\nHybridSearch results:" << std::endl;
    for (auto& result : response.Results().Results()) {
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

    milvus::ConnectParam connect_param{"http://localhost:19530", "root:Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    // collection schema, drop and create collection
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->SetEnableDynamicField(true);
    collection_schema->AddField({field_id, milvus::DataType::INT64, "", true, false});
    collection_schema->AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));
    collection_schema->AddField(milvus::FieldSchema(field_timestamp, milvus::DataType::TIMESTAMPTZ));

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
    insertData(client);

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

    // search/query with timezone
    std::string ts_field_name = field_timestamp;
    const std::vector<std::string> timezones = {"Asia/Shanghai", "America/Havana", "Africa/Bangui", "Australia/Sydney"};
    for (const auto& timezone : timezones) {
        std::cout << "\n================== Query with timezone: " << timezone << " ==================" << std::endl;
        query(client, timezone);
        search(client, timezone);
        hybridSearch(client, timezone);
    }

    client->Disconnect();
    return 0;
}
