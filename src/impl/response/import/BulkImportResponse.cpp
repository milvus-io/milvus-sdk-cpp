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

#include "milvus/response/import/BulkImportResponse.h"

#include <cstdint>
#include <utility>

namespace milvus {

const nlohmann::json&
BulkImportResponse::RawJson() const {
    return json_;
}

void
BulkImportResponse::SetRawJson(nlohmann::json&& json) {
    json_ = std::move(json);
}

int64_t
BulkImportResponse::Code() const {
    auto it = json_.find("code");
    if (it != json_.end() && it->is_number_integer()) {
        return it->get<int64_t>();
    }
    return -1;
}

std::string
BulkImportResponse::Message() const {
    auto it = json_.find("message");
    if (it != json_.end() && it->is_string()) {
        return it->get<std::string>();
    }
    return "";
}

const nlohmann::json&
BulkImportResponse::Data() const {
    static const nlohmann::json empty = nlohmann::json::object();
    auto it = json_.find("data");
    if (it != json_.end() && it->is_object()) {
        return *it;
    }
    return empty;
}

std::string
BulkImportResponse::JobId() const {
    auto it = Data().find("jobId");
    if (it != Data().end() && it->is_string()) {
        return it->get<std::string>();
    }
    return "";
}

}  // namespace milvus
