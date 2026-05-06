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
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {
const char* const collection_name = "cpp_sdk_example_optimize_v2";
const char* const id_field = "id";
const char* const vector_field = "vector";
const uint32_t vector_dim = 512;
const int64_t total_rows = 1000000;
const int64_t batch_size = 10000;

int
printSegmentInfo(milvus::MilvusClientV2Ptr& client) {
    milvus::ListQuerySegmentsResponse response;
    auto status =
        client->ListQuerySegments(milvus::ListQuerySegmentsRequest().WithCollectionName(collection_name), response);
    util::CheckStatus("list query segments", status);

    const auto& segments = response.Result();
    std::cout << "  Total segments: " << segments.size() << std::endl;

    int64_t total_rows_in_segments = 0;
    for (const auto& segment : segments) {
        std::cout << "    Segment " << segment.SegmentID() << ": rows=" << segment.RowCount()
                  << ", state=" << static_cast<int>(segment.State()) << ", index=" << segment.IndexName() << std::endl;
        total_rows_in_segments += segment.RowCount();
    }
    std::cout << "  Total rows across segments: " << total_rows_in_segments << std::endl;
    return static_cast<int>(segments.size());
}

}  // namespace

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClientV2::Create();

    milvus::ConnectParam connect_param{"http://localhost:19530"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    std::cout << "========== Step 1: Create collection ==========" << std::endl;
    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));

    milvus::CollectionSchemaPtr schema = std::make_shared<milvus::CollectionSchema>();
    schema->AddField(milvus::FieldSchema(id_field, milvus::DataType::INT64, "", true, true));
    schema->AddField(milvus::FieldSchema(vector_field, milvus::DataType::FLOAT_VECTOR).WithDimension(vector_dim));

    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(schema));
    util::CheckStatus("create collection: " + std::string(collection_name), status);

    std::cout << "========== Step 2: Insert 1,000,000 rows ==========" << std::endl;
    int64_t total_inserted = 0;
    for (int64_t batch = 0; batch < total_rows / batch_size; ++batch) {
        milvus::EntityRows rows;
        rows.reserve(batch_size);
        for (int64_t i = 0; i < batch_size; ++i) {
            milvus::EntityRow row;
            row[vector_field] = util::GenerateFloatVector(vector_dim);
            rows.emplace_back(std::move(row));
        }

        milvus::InsertResponse insert_response;
        status = client->Insert(
            milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)), insert_response);
        util::CheckStatus("insert", status);

        total_inserted += insert_response.Results().InsertCount();
        if ((batch + 1) % 10 == 0) {
            std::cout << "  Inserted " << total_inserted << " / " << total_rows << " rows" << std::endl;
        }
    }

    status = client->Flush(milvus::FlushRequest().AddCollectionName(collection_name));
    util::CheckStatus("flush", status);
    std::cout << "Total inserted: " << total_inserted << " rows" << std::endl;

    std::cout << "========== Step 3: Create IVF_FLAT index ==========" << std::endl;
    milvus::IndexDesc index(vector_field, "", milvus::IndexType::IVF_FLAT, milvus::MetricType::L2);
    index.AddExtraParam("nlist", "32");
    status = client->CreateIndex(milvus::CreateIndexRequest()
                                     .WithCollectionName(collection_name)
                                     .AddIndex(std::move(index))
                                     .WithTimeoutMs(100000));
    util::CheckStatus("create IVF_FLAT index", status);

    std::cout << "========== Step 4: Load collection ==========" << std::endl;
    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection", status);

    std::cout << "========== Step 5: Query segment info (before optimize) ==========" << std::endl;
    printSegmentInfo(client);

    std::cout << "========== Step 6: Optimize (targetSize=4GB, async) ==========" << std::endl;
    const auto start_time = std::chrono::steady_clock::now();
    milvus::OptimizeTaskPtr task;
    status = client->Optimize(
        milvus::OptimizeRequest().WithCollectionName(collection_name).WithTargetSize("4GB").WithAsync(true), task);
    util::CheckStatus("optimize", status);

    std::string last_progress;
    while (!task->IsDone()) {
        auto progress = task->CurrentProgress();
        if (!progress.empty() && progress != last_progress) {
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time);
            std::cout << "  Optimize progress [" << elapsed.count() << "s]: " << progress << std::endl;
            last_progress = progress;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    milvus::OptimizeResponse optimize_response;
    status = task->GetResult(optimize_response);
    util::CheckStatus("get optimize result", status);

    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time);
    std::cout << "Optimize completed in " << elapsed.count() / 1000.0 << " seconds" << std::endl;
    std::cout << "  Status: " << optimize_response.StatusText() << std::endl;
    std::cout << "  Compaction ID: " << optimize_response.CompactionID() << std::endl;
    std::cout << "  Progress:";
    for (const auto& progress : optimize_response.ProgressHistory()) {
        std::cout << " " << progress;
    }
    std::cout << std::endl;

    std::cout << "========== Step 7: Query segment info (after optimize) ==========" << std::endl;
    const auto step7_start_time = std::chrono::steady_clock::now();
    while (true) {
        int segment_count = printSegmentInfo(client);
        if (segment_count == 1) {
            std::cout << "Optimization successful, only one segment remains" << std::endl;
            break;
        }
        std::cout << "Waiting for optimization to complete..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    const auto step7_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - step7_start_time);
    std::cout << "Step 7 completed in " << step7_elapsed.count() / 1000.0 << " seconds" << std::endl;

    client->Disconnect();
    return 0;
}
