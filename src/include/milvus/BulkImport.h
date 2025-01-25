#pragma once

#include <httplib.h>

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace milvus {

class BulkImport {
 public:
    static nlohmann::json
    CreateImportJobs(const std::string& url, const std::string& collection_name, const std::vector<std::string>& files,
                     const std::string& db_name = "default", const std::string& api_key = "",
                     const std::string& partition_name = "", const nlohmann::json& options = nlohmann::json{});

    static nlohmann::json
    ListImportJobs(const std::string& url, const std::string& collection_name, const std::string& db_name = "default",
                   const std::string& api_key = "");

    static nlohmann::json
    GetImportJobProgress(const std::string& url, const std::string& job_id, const std::string& db_name = "default",
                         const std::string& api_key = "");
};

}  // namespace milvus
