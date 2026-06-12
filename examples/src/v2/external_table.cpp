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

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {

const char* const collection_name = "CPP_V2_EXTERNAL_TABLE";
const char* const field_id = "product_id";
const char* const field_vector = "embedding";
const char* const field_text = "content";
const uint32_t dimension = 128;
const char* const default_external_source = "s3://minio:9000/a-bucket/external_table_example_data/data.parquet";
const char* const default_external_spec = R"({
  "format": "parquet",
  "extfs": {
    "access_key_id": "minioadmin",
    "access_key_value": "minioadmin",
    "region": "us-east-1",
    "use_ssl": "false",
    "use_virtual_host": "false",
    "cloud_provider": "minio"
  }
})";

milvus::RefreshExternalCollectionJobInfo
waitRefreshComplete(milvus::MilvusClientV2Ptr& client, int64_t job_id) {
    for (int i = 0; i < 60; ++i) {
        milvus::GetRefreshExternalCollectionProgressResponse response;
        auto status = client->GetRefreshExternalCollectionProgress(
            milvus::GetRefreshExternalCollectionProgressRequest().WithJobID(job_id), response);
        util::CheckStatus("get refresh external collection progress", status);

        const auto& job_info = response.JobInfo();
        std::cout << "Refresh job " << job_info.JobID() << ": state=" << milvus::to_string(job_info.State())
                  << ", progress=" << job_info.Progress() << "%" << std::endl;
        if (job_info.State() == milvus::RefreshExternalCollectionStateCode::COMPLETED) {
            return job_info;
        }
        if (job_info.State() == milvus::RefreshExternalCollectionStateCode::FAILED) {
            std::cerr << "Refresh external collection failed: " << job_info.Reason() << std::endl;
            exit(1);
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cerr << "Refresh external collection did not complete within 60 seconds" << std::endl;
    exit(1);
}

void
createExternalCollection(milvus::MilvusClientV2Ptr& client, const std::string& external_source) {
    milvus::CollectionSchemaPtr schema = std::make_shared<milvus::CollectionSchema>();
    schema->WithExternalSource(external_source).WithExternalSpec(default_external_spec);
    schema->AddField(milvus::FieldSchema()
                         .WithName(field_id)
                         .WithDataType(milvus::DataType::INT64)
                         .WithExternalField("id"));
    schema->AddField(milvus::FieldSchema()
                         .WithName(field_vector)
                         .WithDataType(milvus::DataType::FLOAT_VECTOR)
                         .WithDimension(dimension)
                         .WithExternalField("vector"));
    schema->AddField(milvus::FieldSchema()
                         .WithName(field_text)
                         .WithDataType(milvus::DataType::VARCHAR)
                         .WithMaxLength(256)
                         .WithExternalField("text"));

    auto status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(milvus::CreateCollectionRequest()
                                          .WithCollectionName(collection_name)
                                          .WithCollectionSchema(schema)
                                          .WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED));
    util::CheckStatus("create collection: " + std::string(collection_name), status);
}

void
verifyData(milvus::MilvusClientV2Ptr& client) {
    auto status = client->CreateIndex(milvus::CreateIndexRequest()
                                          .WithCollectionName(collection_name)
                                          .AddIndex(milvus::IndexDesc(field_vector, "", milvus::IndexType::FLAT,
                                                                      milvus::MetricType::L2)));
    util::CheckStatus("create index on vector field", status);

    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection: " + std::string(collection_name), status);

    milvus::QueryResponse query_response;
    status = client->Query(milvus::QueryRequest()
                               .WithCollectionName(collection_name)
                               .WithLimit(3)
                               .AddOutputField(field_id)
                               .AddOutputField(field_text)
                               .AddOutputField(field_vector)
                               .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG),
                           query_response);
    util::CheckStatus("query external collection", status);

    milvus::EntityRows query_rows;
    status = query_response.Results().OutputRows(query_rows);
    util::CheckStatus("get query output rows", status);

    if (query_rows.empty()) {
        std::cerr << "Query returned no rows after refresh" << std::endl;
        exit(1);
    }

    std::cout << "Query results:" << std::endl;
    for (const auto& row : query_rows) {
        std::cout << "\t" << row << std::endl;
    }

    auto query_vector = query_rows.front()[field_vector].get<std::vector<float>>();

    milvus::SearchResponse search_response;
    status = client->Search(milvus::SearchRequest()
                                .WithCollectionName(collection_name)
                                .WithAnnsField(field_vector)
                                .WithLimit(5)
                                .AddOutputField(field_id)
                                .AddOutputField(field_text)
                                .AddFloatVector(query_vector)
                                .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG),
                            search_response);
    util::CheckStatus("search external collection", status);

    std::cout << "Search results:" << std::endl;
    for (auto& result : search_response.Results().Results()) {
        milvus::EntityRows output_rows;
        status = result.OutputRows(output_rows);
        util::CheckStatus("get search output rows", status);
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

    milvus::ConnectParam connect_param{"http://localhost:19530"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    // Path to the existing Parquet file in the Milvus-configured object storage.
    // The Parquet file is expected to contain source columns `id`, `vector`, and `text`,
    // which are mapped to collection fields `product_id`, `embedding`, and `content`.
    const std::string external_source = argc > 1 ? argv[1] : default_external_source;
    std::cout << "Using external source: " << external_source << std::endl;

    createExternalCollection(client, external_source);

    milvus::DescribeCollectionResponse desc_response;
    status = client->DescribeCollection(milvus::DescribeCollectionRequest().WithCollectionName(collection_name),
                                        desc_response);
    util::CheckStatus("describe collection: " + std::string(collection_name), status);
    std::cout << "Collection external source: " << desc_response.Desc().Schema().ExternalSource() << std::endl;
    std::cout << "Collection external spec: " << desc_response.Desc().Schema().ExternalSpec() << std::endl;

    milvus::RefreshExternalCollectionResponse refresh_response;
    status = client->RefreshExternalCollection(
        milvus::RefreshExternalCollectionRequest().WithCollectionName(collection_name), refresh_response);
    util::CheckStatus("refresh external collection: " + std::string(collection_name), status);
    std::cout << "Refresh job started, jobId=" << refresh_response.JobID() << std::endl;

    auto job_info = waitRefreshComplete(client, refresh_response.JobID());
    std::cout << "Refresh completed: state=" << milvus::to_string(job_info.State())
              << ", progress=" << job_info.Progress() << "%, reason=" << job_info.Reason() << std::endl;

    milvus::ListRefreshExternalCollectionJobsResponse jobs_response;
    status = client->ListRefreshExternalCollectionJobs(
        milvus::ListRefreshExternalCollectionJobsRequest().WithCollectionName(collection_name), jobs_response);
    util::CheckStatus("list refresh external collection jobs", status);
    std::cout << "Refresh jobs:" << std::endl;
    for (const auto& job : jobs_response.Jobs()) {
        std::cout << "\tjobId=" << job.JobID() << ", collection=" << job.CollectionName()
                  << ", state=" << milvus::to_string(job.State()) << ", progress=" << job.Progress()
                  << "%, externalSource=" << job.ExternalSource() << std::endl;
    }

    verifyData(client);

    client->Disconnect();
    return 0;
}
