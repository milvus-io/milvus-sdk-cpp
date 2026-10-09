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

namespace {

struct ImportServer {
    httplib::Server server;
    std::mutex mutex;
    std::vector<std::string> paths;
    std::vector<std::string> bodies;
    std::vector<std::string> authorizations;
    std::vector<std::string> db_names;
    size_t created_jobs = 0;
    int envelope_code = 0;
    std::string envelope_message = "success";
    int port = -1;
    std::thread thread;

    bool
    Start() {
        auto handler = [this](const httplib::Request& request, httplib::Response& response) {
            nlohmann::json payload = {
                {"code", envelope_code},
                {"message", envelope_message},
                {"data", nlohmann::json::object()},
            };
            {
                std::lock_guard<std::mutex> lock(mutex);
                paths.emplace_back(request.path);
                bodies.emplace_back(request.body);
                authorizations.emplace_back(request.get_header_value("Authorization"));
                db_names.emplace_back(request.get_header_value("DB-Name"));
                if (request.path == "/v2/vectordb/jobs/import/create") {
                    payload["data"]["jobId"] = "job-" + std::to_string(++created_jobs);
                }
            }
            response.set_content(payload.dump(), "application/json");
        };
        for (const auto& path : {"/v2/vectordb/jobs/import/create", "/v2/vectordb/jobs/import/list",
                                 "/v2/vectordb/jobs/import/describe", "/v2/vectordb/jobs/import/get_progress",
                                 "/v2/vectordb/jobs/import/commit", "/v2/vectordb/jobs/import/abort"}) {
            server.Post(path, handler);
        }
        port = server.bind_to_any_port("127.0.0.1");
        if (port <= 0) {
            return false;
        }
        thread = std::thread([this]() { server.listen_after_bind(); });
        server.wait_until_ready();
        return server.is_running();
    }

    ~ImportServer() {
        if (server.is_running()) {
            server.stop();
        }
        if (thread.joinable()) {
            thread.join();
        }
    }

    std::string
    Url() const {
        return "http://127.0.0.1:" + std::to_string(port);
    }
};

}  // namespace

TEST(BulkImportTest, DtoCreateMilvusImportJobs) {
    ImportServer server;
    ASSERT_TRUE(server.Start());

    milvus::MilvusImportRequest request;
    request.WithApiKey("token")
        .WithDatabaseName("dto-db")
        .WithCollectionName("collection")
        .WithPartitionName("part")
        .WithFiles({{"a.parquet", "b.parquet"}})
        .WithOptions(nlohmann::json{{"auto_commit", "false"}});
    milvus::BulkImportResponse response;
    ASSERT_TRUE(milvus::BulkImport::CreateImportJobs(server.Url(), request, response).IsOk());
    EXPECT_EQ(response.Code(), 0);
    EXPECT_EQ(response.JobId(), "job-1");

    std::lock_guard<std::mutex> lock(server.mutex);
    ASSERT_EQ(server.paths.size(), 1);
    EXPECT_EQ(server.paths.at(0), "/v2/vectordb/jobs/import/create");
    EXPECT_EQ(server.authorizations.at(0), "Bearer token");
    const auto payload = nlohmann::json::parse(server.bodies.at(0));
    EXPECT_EQ(payload.at("dbName"), "dto-db");
    EXPECT_EQ(payload.at("collectionName"), "collection");
    EXPECT_EQ(payload.at("partitionName"), "part");
    EXPECT_EQ(payload.at("files"), nlohmann::json::array({{"a.parquet", "b.parquet"}}));
    EXPECT_EQ(payload.at("options").at("auto_commit"), "false");
}

TEST(BulkImportTest, DtoCreateVolumeImportJobs) {
    ImportServer server;
    ASSERT_TRUE(server.Start());

    milvus::VolumeImportRequest request;
    request.WithApiKey("token")
        .WithClusterId("cluster-a")
        .WithDatabaseName("db-1")
        .WithCollectionName("collection")
        .WithPartitionName("part-1")
        .WithVolumeName("vol-1")
        .WithDataPaths({{"parquet-folder/"}});
    milvus::BulkImportResponse response;
    ASSERT_TRUE(milvus::BulkImport::CreateImportJobs(server.Url(), request, response).IsOk());

    std::lock_guard<std::mutex> lock(server.mutex);
    ASSERT_EQ(server.paths.size(), 1);
    const auto payload = nlohmann::json::parse(server.bodies.at(0));
    EXPECT_EQ(payload.at("clusterId"), "cluster-a");
    EXPECT_EQ(payload.at("dbName"), "db-1");
    EXPECT_EQ(payload.at("collectionName"), "collection");
    EXPECT_EQ(payload.at("partitionName"), "part-1");
    EXPECT_EQ(payload.at("volumeName"), "vol-1");
    EXPECT_EQ(payload.at("dataPaths"), nlohmann::json::array({{"parquet-folder/"}}));
}

TEST(BulkImportTest, DtoCreateCloudImportJobs) {
    ImportServer server;
    ASSERT_TRUE(server.Start());

    milvus::CloudImportRequest request;
    request.WithApiKey("token")
        .WithClusterId("cluster-a")
        .WithProjectId("p-1")
        .WithRegionId("r-1")
        .WithCollectionName("collection")
        .WithObjectUrls({{"s3://bucket/1.parquet"}})
        .WithAccessKey("ak")
        .WithSecretKey("sk")
        .WithToken("tok");
    milvus::BulkImportResponse response;
    ASSERT_TRUE(milvus::BulkImport::CreateImportJobs(server.Url(), request, response).IsOk());

    std::lock_guard<std::mutex> lock(server.mutex);
    ASSERT_EQ(server.paths.size(), 1);
    const auto payload = nlohmann::json::parse(server.bodies.at(0));
    EXPECT_EQ(payload.at("clusterId"), "cluster-a");
    EXPECT_EQ(payload.at("projectId"), "p-1");
    EXPECT_EQ(payload.at("regionId"), "r-1");
    EXPECT_EQ(payload.at("collectionName"), "collection");
    EXPECT_EQ(payload.at("objectUrls"), nlohmann::json::array({{"s3://bucket/1.parquet"}}));
    EXPECT_EQ(payload.at("accessKey"), "ak");
    EXPECT_EQ(payload.at("secretKey"), "sk");
    EXPECT_EQ(payload.at("token"), "tok");
}

TEST(BulkImportTest, DtoListMilvusImportJobs) {
    ImportServer server;
    ASSERT_TRUE(server.Start());

    milvus::MilvusListImportJobsRequest request;
    request.WithApiKey("token").WithCollectionName("collection").WithDatabaseName("db-1");
    milvus::BulkImportResponse response;
    ASSERT_TRUE(milvus::BulkImport::ListImportJobs(server.Url(), request, response).IsOk());

    std::lock_guard<std::mutex> lock(server.mutex);
    ASSERT_EQ(server.paths.size(), 1);
    EXPECT_EQ(server.paths.at(0), "/v2/vectordb/jobs/import/list");
    const auto payload = nlohmann::json::parse(server.bodies.at(0));
    EXPECT_EQ(payload.at("collectionName"), "collection");
    EXPECT_EQ(payload.at("dbName"), "db-1");
}

TEST(BulkImportTest, DtoListCloudImportJobs) {
    ImportServer server;
    ASSERT_TRUE(server.Start());

    milvus::CloudListImportJobsRequest request;
    request.WithApiKey("token")
        .WithClusterId("cluster-a")
        .WithProjectId("p-1")
        .WithRegionId("r-1")
        .WithPageSize(10)
        .WithCurrentPage(2);
    milvus::BulkImportResponse response;
    ASSERT_TRUE(milvus::BulkImport::ListImportJobs(server.Url(), request, response).IsOk());

    std::lock_guard<std::mutex> lock(server.mutex);
    ASSERT_EQ(server.paths.size(), 1);
    const auto payload = nlohmann::json::parse(server.bodies.at(0));
    EXPECT_EQ(payload.at("clusterId"), "cluster-a");
    EXPECT_EQ(payload.at("projectId"), "p-1");
    EXPECT_EQ(payload.at("regionId"), "r-1");
    EXPECT_EQ(payload.at("pageSize"), 10);
    EXPECT_EQ(payload.at("currentPage"), 2);
}

TEST(BulkImportTest, DtoDescribeCommitAndAbortImport) {
    ImportServer server;
    ASSERT_TRUE(server.Start());

    milvus::MilvusDescribeImportRequest describe;
    describe.WithApiKey("token").WithDatabaseName("db-1").WithJobId("123");
    milvus::BulkImportResponse response;
    ASSERT_TRUE(milvus::BulkImport::GetImportJobProgress(server.Url(), describe, response).IsOk());

    milvus::MilvusDescribeImportRequest commit;
    commit.WithApiKey("token").WithDatabaseName("db-1").WithJobId("456");
    ASSERT_TRUE(milvus::BulkImport::CommitImport(server.Url(), commit, response).IsOk());

    milvus::MilvusDescribeImportRequest abort;
    abort.WithApiKey("token").WithDatabaseName("db-1").WithJobId("789");
    ASSERT_TRUE(milvus::BulkImport::AbortImport(server.Url(), abort, response).IsOk());

    std::lock_guard<std::mutex> lock(server.mutex);
    ASSERT_EQ(server.paths.size(), 3);
    EXPECT_EQ(server.paths.at(0), "/v2/vectordb/jobs/import/describe");
    EXPECT_EQ(server.paths.at(1), "/v2/vectordb/jobs/import/commit");
    EXPECT_EQ(server.paths.at(2), "/v2/vectordb/jobs/import/abort");
    for (const auto& db_name : server.db_names) {
        EXPECT_EQ(db_name, "db-1");
    }

    auto payload = nlohmann::json::parse(server.bodies.at(0));
    EXPECT_EQ(payload.at("jobId"), "123");
    EXPECT_FALSE(payload.contains("jobID"));
    payload = nlohmann::json::parse(server.bodies.at(1));
    EXPECT_EQ(payload.at("jobId"), "456");
    payload = nlohmann::json::parse(server.bodies.at(2));
    EXPECT_EQ(payload.at("jobId"), "789");
}

TEST(BulkImportTest, LegacyGetImportJobProgressUsesJobIdField) {
    ImportServer server;
    ASSERT_TRUE(server.Start());

    const auto response = milvus::BulkImport::GetImportJobProgress(server.Url(), "123", "legacy-db", "token");
    ASSERT_FALSE(response.is_null());

    std::lock_guard<std::mutex> lock(server.mutex);
    ASSERT_EQ(server.paths.size(), 1);
    EXPECT_EQ(server.paths.at(0), "/v2/vectordb/jobs/import/get_progress");
    const auto payload = nlohmann::json::parse(server.bodies.at(0));
    EXPECT_EQ(payload.at("jobId"), "123");
    EXPECT_FALSE(payload.contains("jobID"));
    EXPECT_EQ(server.db_names.at(0), "legacy-db");
}

TEST(BulkImportTest, DtoCloudDescribeImport) {
    ImportServer server;
    ASSERT_TRUE(server.Start());

    milvus::CloudDescribeImportRequest request;
    request.WithApiKey("token").WithClusterId("cluster-a").WithProjectId("p-1").WithRegionId("r-1").WithJobId("123");
    milvus::BulkImportResponse response;
    ASSERT_TRUE(milvus::BulkImport::GetImportJobProgress(server.Url(), request, response).IsOk());

    std::lock_guard<std::mutex> lock(server.mutex);
    ASSERT_EQ(server.paths.size(), 1);
    const auto payload = nlohmann::json::parse(server.bodies.at(0));
    EXPECT_EQ(payload.at("clusterId"), "cluster-a");
    EXPECT_EQ(payload.at("projectId"), "p-1");
    EXPECT_EQ(payload.at("regionId"), "r-1");
    EXPECT_EQ(payload.at("jobId"), "123");
}

TEST(BulkImportTest, DtoImportReportsServerEnvelopeError) {
    ImportServer server;
    server.envelope_code = 1100;
    server.envelope_message = "out of memory";
    ASSERT_TRUE(server.Start());

    milvus::MilvusImportRequest request;
    request.WithApiKey("token").WithCollectionName("collection").WithFiles({{"a.parquet"}});
    milvus::BulkImportResponse response;
    const auto status = milvus::BulkImport::CreateImportJobs(server.Url(), request, response);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(status.Code(), milvus::StatusCode::SERVER_FAILED);
    EXPECT_NE(status.Message().find("1100"), std::string::npos);
    EXPECT_NE(status.Message().find("out of memory"), std::string::npos);
    EXPECT_EQ(response.Code(), 1100);
}

TEST(BulkImportTest, DtoCreateImportWithConfig) {
    ImportServer server;
    ASSERT_TRUE(server.Start());

    milvus::MilvusImportRequest request;
    request.WithApiKey("token").WithCollectionName("collection").WithFiles({{"a.parquet"}});
    milvus::BulkImportConfig config;
    config.WithTimeout(5).WithVerifyServerCert(false);
    milvus::BulkImportResponse response;
    ASSERT_TRUE(milvus::BulkImport::CreateImportJobs(server.Url(), request, response, config).IsOk());
    EXPECT_EQ(response.JobId(), "job-1");
}
