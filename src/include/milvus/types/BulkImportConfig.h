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

#include <string>

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Transport options for the bulk import REST API calls.
 * Client certificates (mutual TLS) are not supported.
 */
class MILVUS_SDK_API BulkImportConfig {
 public:
    /**
     * @brief Constructor
     */
    BulkImportConfig() = default;

    /**
     * @brief Get the request timeout in seconds, zero keeps the httplib default.
     * @return the timeout.
     */
    int64_t
    Timeout() const;

    /**
     * @brief Set the request timeout in seconds.
     * @param [in] timeout the timeout in seconds.
     */
    void
    SetTimeout(int64_t timeout);

    /**
     * @brief Set the request timeout in seconds.
     * @param [in] timeout the timeout in seconds.
     */
    BulkImportConfig&
    WithTimeout(int64_t timeout);

    /**
     * @brief Get whether the server certificate is verified for https calls.
     * @return true if the certificate is verified.
     */
    bool
    VerifyServerCert() const;

    /**
     * @brief Set whether the server certificate is verified for https calls.
     * @param [in] verify whether to verify the certificate.
     */
    void
    SetVerifyServerCert(bool verify);

    /**
     * @brief Set whether the server certificate is verified for https calls.
     * @param [in] verify whether to verify the certificate.
     */
    BulkImportConfig&
    WithVerifyServerCert(bool verify);

    /**
     * @brief Get the path of the CA certificate bundle used to verify the server.
     * @return the CA certificate path.
     */
    const std::string&
    CaCertPath() const;

    /**
     * @brief Set the path of the CA certificate bundle used to verify the server.
     * @param [in] ca_cert_path the CA certificate path.
     */
    void
    SetCaCertPath(const std::string& ca_cert_path);

    /**
     * @brief Set the path of the CA certificate bundle used to verify the server.
     * @param [in] ca_cert_path the CA certificate path.
     */
    BulkImportConfig&
    WithCaCertPath(const std::string& ca_cert_path);

 private:
    int64_t timeout_{0};
    bool verify_server_cert_{true};
    std::string ca_cert_path_;
};

}  // namespace milvus
