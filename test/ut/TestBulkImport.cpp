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

#include <cpp-httplib/httplib.h>
#include <gtest/gtest.h>

#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "milvus/BulkImport.h"

TEST(BulkImportTest, CreateCommitAndAbortImport) {
    httplib::Server server;
    std::mutex requests_mutex;
    std::vector<std::string> request_paths;
    std::vector<std::string> request_bodies;
    std::vector<std::string> authorization_headers;
    size_t created_jobs = 0;

    auto handler = [&](const httplib::Request& request, httplib::Response& response) {
        nlohmann::json response_payload = {
            {"code", 0},
            {"message", "success"},
            {"data", nlohmann::json::object()},
        };
        {
            std::lock_guard<std::mutex> lock(requests_mutex);
            request_paths.emplace_back(request.path);
            request_bodies.emplace_back(request.body);
            authorization_headers.emplace_back(request.get_header_value("Authorization"));
            if (request.path == "/v2/vectordb/jobs/import/create") {
                response_payload["data"]["jobId"] = created_jobs++ == 0 ? "123" : "456";
            }
        }
        response.set_content(response_payload.dump(), "application/json");
    };
    server.Post("/v2/vectordb/jobs/import/create", handler);
    server.Post("/v2/vectordb/jobs/import/commit", handler);
    server.Post("/v2/vectordb/jobs/import/abort", handler);

    const auto port = server.bind_to_any_port("127.0.0.1");
    ASSERT_GT(port, 0);
    std::thread server_thread([&]() { server.listen_after_bind(); });
    server.wait_until_ready();
    if (!server.is_running()) {
        server_thread.join();
        FAIL() << "Failed to start the test HTTP server";
    }

    const auto url = "http://127.0.0.1:" + std::to_string(port);
    const nlohmann::json options = {{"auto_commit", "false"}};
    auto create_commit_response =
        milvus::BulkImport::CreateImportJobs(url, "collection", {"commit.parquet"}, "commit-db", "token", "", options);
    auto create_abort_response =
        milvus::BulkImport::CreateImportJobs(url, "collection", {"abort.parquet"}, "abort-db", "token", "", options);
    if (create_commit_response.is_null() || create_abort_response.is_null()) {
        server.stop();
        server_thread.join();
        FAIL() << "Failed to create the test import jobs";
    }
    auto commit_response = milvus::BulkImport::CommitImport(
        url, create_commit_response.at("data").at("jobId").get<std::string>(), "commit-db", "token");
    auto abort_response = milvus::BulkImport::AbortImport(
        url, create_abort_response.at("data").at("jobId").get<std::string>(), "abort-db", "token");

    server.stop();
    server_thread.join();

    ASSERT_FALSE(create_commit_response.is_null());
    ASSERT_FALSE(create_abort_response.is_null());
    ASSERT_FALSE(commit_response.is_null());
    ASSERT_FALSE(abort_response.is_null());
    EXPECT_EQ(commit_response.at("code").get<int>(), 0);
    EXPECT_EQ(abort_response.at("code").get<int>(), 0);

    std::lock_guard<std::mutex> lock(requests_mutex);
    ASSERT_EQ(request_paths.size(), 4);
    ASSERT_EQ(request_bodies.size(), 4);
    ASSERT_EQ(authorization_headers.size(), 4);
    EXPECT_EQ(request_paths.at(0), "/v2/vectordb/jobs/import/create");
    EXPECT_EQ(request_paths.at(1), "/v2/vectordb/jobs/import/create");
    EXPECT_EQ(request_paths.at(2), "/v2/vectordb/jobs/import/commit");
    EXPECT_EQ(request_paths.at(3), "/v2/vectordb/jobs/import/abort");
    for (const auto& authorization : authorization_headers) {
        EXPECT_EQ(authorization, "Bearer token");
    }

    const auto create_commit_payload = nlohmann::json::parse(request_bodies.at(0));
    EXPECT_EQ(create_commit_payload.at("dbName"), "commit-db");
    EXPECT_EQ(create_commit_payload.at("options").at("auto_commit"), "false");
    EXPECT_FALSE(create_commit_payload.at("options").contains("timeout"));

    const auto create_abort_payload = nlohmann::json::parse(request_bodies.at(1));
    EXPECT_EQ(create_abort_payload.at("dbName"), "abort-db");
    EXPECT_EQ(create_abort_payload.at("options").at("auto_commit"), "false");
    EXPECT_FALSE(create_abort_payload.at("options").contains("timeout"));

    const auto commit_payload = nlohmann::json::parse(request_bodies.at(2));
    EXPECT_EQ(commit_payload.at("dbName"), "commit-db");
    EXPECT_EQ(commit_payload.at("jobId"), "123");
    EXPECT_FALSE(commit_payload.contains("jobID"));

    const auto abort_payload = nlohmann::json::parse(request_bodies.at(3));
    EXPECT_EQ(abort_payload.at("dbName"), "abort-db");
    EXPECT_EQ(abort_payload.at("jobId"), "456");
    EXPECT_FALSE(abort_payload.contains("jobID"));
}
