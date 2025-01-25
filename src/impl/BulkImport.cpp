#include "milvus/BulkImport.h"

namespace milvus {

nlohmann::json
BulkImport::CreateImportJobs(const std::string& url, const std::string& collection_name,
                             const std::vector<std::string>& files, const std::string& db_name,
                             const std::string& api_key, const std::string& partition_name,
                             const nlohmann::json& options) {
    httplib::Client client(url);

    nlohmann::json request_payload = {
        {"dbName", db_name},
        {"collectionName", collection_name},
        {"files", nlohmann::json::array({files})},
    };

    if (!partition_name.empty()) {
        request_payload["partitionName"] = partition_name;
    }

    if (!options.empty()) {
        if (options.contains("timeout") && !options["timeout"].is_null()) {
            request_payload["options"] = options;
        }
    }

    std::string request_url = "/v2/vectordb/jobs/import/create";
    httplib::Headers headers = {
        {"Authorization", "Bearer " + api_key},
    };
    std::string body = request_payload.dump();

    auto res = client.Post(request_url, headers, body, "application/json");
    if (res && res->status == 200) {
        return nlohmann::json::parse(res->body);
    } else {
        return nullptr;
    }
}

nlohmann::json
BulkImport::ListImportJobs(const std::string& url, const std::string& collection_name, const std::string& db_name,
                           const std::string& api_key) {
    httplib::Client client(url);

    nlohmann::json request_payload = {
        {"collectionName", collection_name},
        {"dbName", db_name},
    };

    std::string request_url = "/v2/vectordb/jobs/import/list";
    httplib::Headers headers = {
        {"Authorization", "Bearer " + api_key},
    };
    std::string body = request_payload.dump();

    auto res = client.Post(request_url, headers, body, "application/json");
    if (res && res->status == 200) {
        return nlohmann::json::parse(res->body);
    } else {
        return nullptr;
    }
}

nlohmann::json
BulkImport::GetImportJobProgress(const std::string& url, const std::string& job_id, const std::string& db_name,
                                 const std::string& api_key) {
    httplib::Client client(url);

    nlohmann::json payload = {{"dbName", db_name}, {"jobID", job_id}};

    std::string request_url = "/v2/vectordb/jobs/import/get_progress";
    httplib::Headers headers = {
        {"Authorization", "Bearer " + api_key},
    };
    std::string body = payload.dump();

    auto res = client.Post(request_url, headers, body, "application/json");
    if (res && res->status == 200) {
        return nlohmann::json::parse(res->body);
    } else {
        return nullptr;
    }
}

}  // namespace milvus
