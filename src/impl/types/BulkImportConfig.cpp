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

#include "milvus/types/BulkImportConfig.h"

namespace milvus {

int64_t
BulkImportConfig::Timeout() const {
    return timeout_;
}

void
BulkImportConfig::SetTimeout(int64_t timeout) {
    timeout_ = timeout;
}

BulkImportConfig&
BulkImportConfig::WithTimeout(int64_t timeout) {
    SetTimeout(timeout);
    return *this;
}

bool
BulkImportConfig::VerifyServerCert() const {
    return verify_server_cert_;
}

void
BulkImportConfig::SetVerifyServerCert(bool verify) {
    verify_server_cert_ = verify;
}

BulkImportConfig&
BulkImportConfig::WithVerifyServerCert(bool verify) {
    SetVerifyServerCert(verify);
    return *this;
}

const std::string&
BulkImportConfig::CaCertPath() const {
    return ca_cert_path_;
}

void
BulkImportConfig::SetCaCertPath(const std::string& ca_cert_path) {
    ca_cert_path_ = ca_cert_path;
}

BulkImportConfig&
BulkImportConfig::WithCaCertPath(const std::string& ca_cert_path) {
    SetCaCertPath(ca_cert_path);
    return *this;
}

}  // namespace milvus
