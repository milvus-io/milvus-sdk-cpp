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
#include <vector>

#include "./BaseImportRequest.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Request for importing data into an open-source Milvus instance.
 */
class MILVUS_SDK_API MilvusImportRequest : public BaseImportRequest<MilvusImportRequest> {
 public:
    /**
     * @brief Constructor
     */
    MilvusImportRequest() = default;

    /**
     * @brief Get the data files to import.
     * @return the files.
     */
    const std::vector<std::vector<std::string>>&
    Files() const;

    /**
     * @brief Set the data files to import.
     * @param [in] files the files.
     */
    void
    SetFiles(std::vector<std::vector<std::string>>&& files);

    /**
     * @brief Set the data files to import.
     * @param [in] files the files.
     */
    MilvusImportRequest&
    WithFiles(std::vector<std::vector<std::string>>&& files);

    /**
     * @brief Serialize the request into the REST import job payload.
     * @return the payload.
     */
    nlohmann::json
    ToJson() const override;

 private:
    std::vector<std::vector<std::string>> files_;
};

}  // namespace milvus
