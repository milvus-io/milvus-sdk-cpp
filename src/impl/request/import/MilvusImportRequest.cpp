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

#include "milvus/request/import/MilvusImportRequest.h"

#include <utility>

namespace milvus {

const std::vector<std::vector<std::string>>&
MilvusImportRequest::Files() const {
    return files_;
}

void
MilvusImportRequest::SetFiles(std::vector<std::vector<std::string>>&& files) {
    files_ = std::move(files);
}

MilvusImportRequest&
MilvusImportRequest::WithFiles(std::vector<std::vector<std::string>>&& files) {
    SetFiles(std::move(files));
    return *this;
}

nlohmann::json
MilvusImportRequest::ToJson() const {
    auto payload = BaseImportRequest::ToJson();
    payload["collectionName"] = collection_name_;
    if (!db_name_.empty()) {
        payload["dbName"] = db_name_;
    }
    if (!partition_name_.empty()) {
        payload["partitionName"] = partition_name_;
    }
    payload["files"] = files_;
    return payload;
}

}  // namespace milvus
