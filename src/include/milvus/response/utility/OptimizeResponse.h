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

#include <cstdint>
#include <string>
#include <vector>

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Result of MilvusClientV2::Optimize().
 */
class MILVUS_SDK_API OptimizeResponse {
 public:
    /**
     * @brief Constructor
     */
    OptimizeResponse() = default;

    /**
     * @brief Get status text.
     */
    const std::string&
    StatusText() const;

    /**
     * @brief Set status text.
     */
    void
    SetStatusText(const std::string& status);

    /**
     * @brief Get collection name.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set collection name.
     */
    void
    SetCollectionName(const std::string& collection_name);

    /**
     * @brief Get compaction ID.
     */
    int64_t
    CompactionID() const;

    /**
     * @brief Set compaction ID.
     */
    void
    SetCompactionID(int64_t compaction_id);

    /**
     * @brief Get normalized target size.
     */
    const std::string&
    TargetSize() const;

    /**
     * @brief Set normalized target size.
     */
    void
    SetTargetSize(const std::string& target_size);

    /**
     * @brief Get progress history.
     */
    const std::vector<std::string>&
    ProgressHistory() const;

    /**
     * @brief Set progress history.
     */
    void
    SetProgressHistory(std::vector<std::string>&& progress_history);

    /**
     * @brief Add progress message.
     */
    void
    AddProgress(const std::string& progress);

 private:
    std::string status_;
    std::string collection_name_;
    int64_t compaction_id_{0};
    std::string target_size_;
    std::vector<std::string> progress_history_;
};

}  // namespace milvus
