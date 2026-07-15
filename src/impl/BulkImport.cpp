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

nlohmann::json
PostImportRequest(const std::string& url, const std::string& request_path, const std::string& api_key,
                  const nlohmann::json& request_payload) {
    httplib::Client client(url);
    httplib::Headers headers = {
        {"Authorization", "Bearer " + api_key},
    };

    auto response = client.Post(request_path, headers, request_payload.dump(), "application/json");
    if (response && response->status == 200) {
        return nlohmann::json::parse(response->body);
    }
    return nullptr;
}

}  // namespace

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
    return PostImportRequest(url, "/v2/vectordb/jobs/import/create", api_key, request_payload);
}

nlohmann::json
BulkImport::ListImportJobs(const std::string& url, const std::string& collection_name, const std::string& db_name,
                           const std::string& api_key) {
    nlohmann::json request_payload = {
        {"collectionName", collection_name},
        {"dbName", db_name},
    };
    return PostImportRequest(url, "/v2/vectordb/jobs/import/list", api_key, request_payload);
}

nlohmann::json
BulkImport::GetImportJobProgress(const std::string& url, const std::string& job_id, const std::string& db_name,
                                 const std::string& api_key) {
    nlohmann::json payload = {{"dbName", db_name}, {"jobID", job_id}};
    return PostImportRequest(url, "/v2/vectordb/jobs/import/get_progress", api_key, payload);
}

nlohmann::json
BulkImport::CommitImport(const std::string& url, const std::string& job_id, const std::string& db_name,
                         const std::string& api_key) {
    nlohmann::json payload = {{"dbName", db_name}, {"jobId", job_id}};
    return PostImportRequest(url, "/v2/vectordb/jobs/import/commit", api_key, payload);
}

nlohmann::json
BulkImport::AbortImport(const std::string& url, const std::string& job_id, const std::string& db_name,
                        const std::string& api_key) {
    nlohmann::json payload = {{"dbName", db_name}, {"jobId", job_id}};
    return PostImportRequest(url, "/v2/vectordb/jobs/import/abort", api_key, payload);
}

}  // namespace milvus
