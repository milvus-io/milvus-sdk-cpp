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

#pragma once

#include <milvus/thirdparty/nlohmann/json.hpp>
#include <string>
#include <vector>

#include "milvus/Export.h"
#include "milvus/Status.h"
#include "milvus/request/import/BaseDescribeImportRequest.h"
#include "milvus/request/import/BaseImportRequest.h"
#include "milvus/request/import/BaseListImportJobsRequest.h"
#include "milvus/request/import/CloudDescribeImportRequest.h"
#include "milvus/request/import/CloudImportRequest.h"
#include "milvus/request/import/CloudListImportJobsRequest.h"
#include "milvus/request/import/MilvusDescribeImportRequest.h"
#include "milvus/request/import/MilvusImportRequest.h"
#include "milvus/request/import/MilvusListImportJobsRequest.h"
#include "milvus/request/import/VolumeImportRequest.h"
#include "milvus/response/import/BulkImportResponse.h"
#include "milvus/types/BulkImportConfig.h"

namespace milvus {

class MILVUS_SDK_API BulkImport {
 public:
    /**
     * @brief Create an import job by restful api.
     * @deprecated Use the request/response overload taking a BaseImportRequest instead.
     */
    [[deprecated("Use the request/response overload instead")]] static nlohmann::json
    CreateImportJobs(const std::string& url, const std::string& collection_name, const std::vector<std::string>& files,
                     const std::string& db_name = "default", const std::string& api_key = "",
                     const std::string& partition_name = "", const nlohmann::json& options = nlohmann::json{});

    /**
     * @brief Create an import job by restful api.
     * @param [in] url the platform endpoint.
     * @param [in] request the import request.
     * @param [in] config the transport options.
     * @param [out] response the response.
     */
    template <typename T>
    static Status
    CreateImportJobs(const std::string& url, const BaseImportRequest<T>& request, BulkImportResponse& response,
                     const BulkImportConfig& config = BulkImportConfig{}) {
        return CreateImportJobsImpl(url, request.ApiKey(), request.ToJson(), config, response);
    }

    /**
     * @brief List all import jobs by restful api.
     * @deprecated Use the request/response overload taking a BaseListImportJobsRequest instead.
     */
    [[deprecated("Use the request/response overload instead")]] static nlohmann::json
    ListImportJobs(const std::string& url, const std::string& collection_name, const std::string& db_name = "default",
                   const std::string& api_key = "");

    /**
     * @brief List all import jobs by restful api.
     * @param [in] url the platform endpoint.
     * @param [in] request the list import jobs request.
     * @param [in] config the transport options.
     * @param [out] response the response.
     */
    template <typename T>
    static Status
    ListImportJobs(const std::string& url, const BaseListImportJobsRequest<T>& request, BulkImportResponse& response,
                   const BulkImportConfig& config = BulkImportConfig{}) {
        return ListImportJobsImpl(url, request.ApiKey(), request.ToJson(), config, response);
    }

    /**
     * @brief Get import job progress by restful api.
     * @deprecated Use the request/response overload taking a BaseDescribeImportRequest instead.
     */
    [[deprecated("Use the request/response overload instead")]] static nlohmann::json
    GetImportJobProgress(const std::string& url, const std::string& job_id, const std::string& db_name = "default",
                         const std::string& api_key = "");

    /**
     * @brief Get import job progress by restful api.
     * @param [in] url the platform endpoint.
     * @param [in] request the describe import request.
     * @param [in] config the transport options.
     * @param [out] response the response.
     */
    template <typename T>
    static Status
    GetImportJobProgress(const std::string& url, const BaseDescribeImportRequest<T>& request,
                         BulkImportResponse& response, const BulkImportConfig& config = BulkImportConfig{}) {
        return GetImportJobProgressImpl(url, request.ApiKey(), request.ToJson(), request.DatabaseName(), config,
                                        response);
    }

    /**
     * @brief Commit a 2PC import job created with options.auto_commit=false, making its staged imported data visible.
     * @deprecated Use the request/response overload taking a BaseDescribeImportRequest instead.
     */
    [[deprecated("Use the request/response overload instead")]] static nlohmann::json
    CommitImport(const std::string& url, const std::string& job_id, const std::string& db_name = "default",
                 const std::string& api_key = "");

    /**
     * @brief Commit a 2PC import job created with options.auto_commit=false, making its staged imported data visible.
     * @param [in] url the platform endpoint.
     * @param [in] request the commit import request (a BaseDescribeImportRequest-derived DTO).
     * @param [in] config the transport options.
     * @param [out] response the response.
     */
    template <typename T>
    static Status
    CommitImport(const std::string& url, const BaseDescribeImportRequest<T>& request, BulkImportResponse& response,
                 const BulkImportConfig& config = BulkImportConfig{}) {
        return CommitImportImpl(url, request.ApiKey(), request.ToJson(), request.DatabaseName(), config, response);
    }

    /**
     * @brief Abort a 2PC import job created with options.auto_commit=false, discarding its staged imported data.
     * @deprecated Use the request/response overload taking a BaseDescribeImportRequest instead.
     */
    [[deprecated("Use the request/response overload instead")]] static nlohmann::json
    AbortImport(const std::string& url, const std::string& job_id, const std::string& db_name = "default",
                const std::string& api_key = "");

    /**
     * @brief Abort a 2PC import job created with options.auto_commit=false, discarding its staged imported data.
     * @param [in] url the platform endpoint.
     * @param [in] request the abort import request (a BaseDescribeImportRequest-derived DTO).
     * @param [in] config the transport options.
     * @param [out] response the response.
     */
    template <typename T>
    static Status
    AbortImport(const std::string& url, const BaseDescribeImportRequest<T>& request, BulkImportResponse& response,
                const BulkImportConfig& config = BulkImportConfig{}) {
        return AbortImportImpl(url, request.ApiKey(), request.ToJson(), request.DatabaseName(), config, response);
    }

 private:
    static Status
    CreateImportJobsImpl(const std::string& url, const std::string& api_key, const nlohmann::json& request_payload,
                         const BulkImportConfig& config, BulkImportResponse& response);

    static Status
    ListImportJobsImpl(const std::string& url, const std::string& api_key, const nlohmann::json& request_payload,
                       const BulkImportConfig& config, BulkImportResponse& response);

    static Status
    GetImportJobProgressImpl(const std::string& url, const std::string& api_key, const nlohmann::json& request_payload,
                             const std::string& db_name, const BulkImportConfig& config, BulkImportResponse& response);

    static Status
    CommitImportImpl(const std::string& url, const std::string& api_key, const nlohmann::json& request_payload,
                     const std::string& db_name, const BulkImportConfig& config, BulkImportResponse& response);

    static Status
    AbortImportImpl(const std::string& url, const std::string& api_key, const nlohmann::json& request_payload,
                    const std::string& db_name, const BulkImportConfig& config, BulkImportResponse& response);
};

}  // namespace milvus
