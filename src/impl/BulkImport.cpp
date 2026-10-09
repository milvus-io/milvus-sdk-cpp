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

#include "milvus/BulkImport.h"

#include <cpp-httplib/httplib.h>

namespace milvus {
namespace {

Status
PostImport(const std::string& url, const std::string& request_path, const std::string& api_key,
           const nlohmann::json& request_payload, const std::string& db_name, const BulkImportConfig& config,
           BulkImportResponse& response) {
    response.SetRawJson(nlohmann::json::object());
    try {
        httplib::Client client(url);
        client.enable_server_certificate_verification(config.VerifyServerCert());
        if (!config.CaCertPath().empty()) {
            client.set_ca_cert_path(config.CaCertPath());
        }
        if (config.Timeout() > 0) {
            client.set_connection_timeout(config.Timeout());
            client.set_read_timeout(config.Timeout(), 0);
            client.set_write_timeout(config.Timeout(), 0);
        }
        httplib::Headers headers = {
            {"Authorization", "Bearer " + api_key},
        };
        if (!db_name.empty()) {
            headers.emplace("DB-Name", db_name);
        }

        auto result = client.Post(request_path, headers, request_payload.dump(), "application/json");
        if (!result) {
            return {StatusCode::RPC_FAILED, "failed to post import request: " + httplib::to_string(result.error())};
        }
        if (result->status != 200) {
            return {StatusCode::SERVER_FAILED,
                    "import request failed with HTTP status " + std::to_string(result->status) + ": " + result->body};
        }
        try {
            response.SetRawJson(nlohmann::json::parse(result->body));
        } catch (const std::exception& e) {
            return {StatusCode::JSON_PARSE_ERROR, std::string("failed to parse import response: ") + e.what()};
        }
    } catch (const std::exception& e) {
        return {StatusCode::UNKNOWN_ERROR, std::string("failed to post import request: ") + e.what()};
    }
    return Status::OK();
}

Status
CheckImportResponse(const BulkImportResponse& response) {
    if (response.Code() != 0) {
        return {StatusCode::SERVER_FAILED,
                "import request failed with code " + std::to_string(response.Code()) + ": " + response.Message()};
    }
    return Status::OK();
}

}  // namespace

Status
BulkImport::CreateImportJobsImpl(const std::string& url, const std::string& api_key,
                                 const nlohmann::json& request_payload, const BulkImportConfig& config,
                                 BulkImportResponse& response) {
    auto status = PostImport(url, "/v2/vectordb/jobs/import/create", api_key, request_payload, "", config, response);
    return status.IsOk() ? CheckImportResponse(response) : status;
}

Status
BulkImport::ListImportJobsImpl(const std::string& url, const std::string& api_key,
                               const nlohmann::json& request_payload, const BulkImportConfig& config,
                               BulkImportResponse& response) {
    auto status = PostImport(url, "/v2/vectordb/jobs/import/list", api_key, request_payload, "", config, response);
    return status.IsOk() ? CheckImportResponse(response) : status;
}

Status
BulkImport::GetImportJobProgressImpl(const std::string& url, const std::string& api_key,
                                     const nlohmann::json& request_payload, const std::string& db_name,
                                     const BulkImportConfig& config, BulkImportResponse& response) {
    auto status =
        PostImport(url, "/v2/vectordb/jobs/import/describe", api_key, request_payload, db_name, config, response);
    return status.IsOk() ? CheckImportResponse(response) : status;
}

Status
BulkImport::CommitImportImpl(const std::string& url, const std::string& api_key, const nlohmann::json& request_payload,
                             const std::string& db_name, const BulkImportConfig& config, BulkImportResponse& response) {
    auto status =
        PostImport(url, "/v2/vectordb/jobs/import/commit", api_key, request_payload, db_name, config, response);
    return status.IsOk() ? CheckImportResponse(response) : status;
}

Status
BulkImport::AbortImportImpl(const std::string& url, const std::string& api_key, const nlohmann::json& request_payload,
                            const std::string& db_name, const BulkImportConfig& config, BulkImportResponse& response) {
    auto status =
        PostImport(url, "/v2/vectordb/jobs/import/abort", api_key, request_payload, db_name, config, response);
    return status.IsOk() ? CheckImportResponse(response) : status;
}

nlohmann::json
BulkImport::CreateImportJobs(const std::string& url, const std::string& collection_name,
                             const std::vector<std::string>& files, const std::string& db_name,
                             const std::string& api_key, const std::string& partition_name,
                             const nlohmann::json& options) {
    nlohmann::json request_payload = {
        {"dbName", db_name},
        {"collectionName", collection_name},
        {"files", nlohmann::json::array({files})},
    };

    if (!partition_name.empty()) {
        request_payload["partitionName"] = partition_name;
    }

    if (!options.empty()) {
        request_payload["options"] = options;
    }
    BulkImportResponse response;
    auto status =
        PostImport(url, "/v2/vectordb/jobs/import/create", api_key, request_payload, "", BulkImportConfig{}, response);
    return status.IsOk() ? response.RawJson() : nlohmann::json{};
}

nlohmann::json
BulkImport::ListImportJobs(const std::string& url, const std::string& collection_name, const std::string& db_name,
                           const std::string& api_key) {
    nlohmann::json request_payload = {
        {"collectionName", collection_name},
        {"dbName", db_name},
    };
    BulkImportResponse response;
    auto status =
        PostImport(url, "/v2/vectordb/jobs/import/list", api_key, request_payload, "", BulkImportConfig{}, response);
    return status.IsOk() ? response.RawJson() : nlohmann::json{};
}

nlohmann::json
BulkImport::GetImportJobProgress(const std::string& url, const std::string& job_id, const std::string& db_name,
                                 const std::string& api_key) {
    nlohmann::json payload = {{"dbName", db_name}, {"jobId", job_id}};
    BulkImportResponse response;
    auto status = PostImport(url, "/v2/vectordb/jobs/import/get_progress", api_key, payload, db_name,
                             BulkImportConfig{}, response);
    return status.IsOk() ? response.RawJson() : nlohmann::json{};
}

nlohmann::json
BulkImport::CommitImport(const std::string& url, const std::string& job_id, const std::string& db_name,
                         const std::string& api_key) {
    nlohmann::json payload = {{"dbName", db_name}, {"jobId", job_id}};
    BulkImportResponse response;
    auto status =
        PostImport(url, "/v2/vectordb/jobs/import/commit", api_key, payload, db_name, BulkImportConfig{}, response);
    return status.IsOk() ? response.RawJson() : nlohmann::json{};
}

nlohmann::json
BulkImport::AbortImport(const std::string& url, const std::string& job_id, const std::string& db_name,
                        const std::string& api_key) {
    nlohmann::json payload = {{"dbName", db_name}, {"jobId", job_id}};
    BulkImportResponse response;
    auto status =
        PostImport(url, "/v2/vectordb/jobs/import/abort", api_key, payload, db_name, BulkImportConfig{}, response);
    return status.IsOk() ? response.RawJson() : nlohmann::json{};
}

}  // namespace milvus
