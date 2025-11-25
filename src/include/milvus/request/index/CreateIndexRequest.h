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

#include "../../types/IndexDesc.h"
#include "./IndexRequestBase.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::CreateIndex()
 */
class CreateIndexRequest : public IndexRequestBase {
 public:
    /**
     * @brief Constructor
     */
    CreateIndexRequest() = default;

    /**
     * @brief Set database name in which the collection is created.
     */
    CreateIndexRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Set name of the collection.
     */
    CreateIndexRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Get indexes.
     */
    const std::vector<IndexDesc>&
    Indexes() const;

    /**
     * @brief Set indexes to be created.
     */
    void
    SetIndexes(std::vector<IndexDesc>&& indexes);

    /**
     * @brief Set indexes to be created.
     */
    CreateIndexRequest&
    WithIndexes(std::vector<IndexDesc>&& indexes);

    /**
     * @brief Add an index to be created.
     */
    CreateIndexRequest&
    AddIndex(IndexDesc&& index);

    /**
     * @brief Get sync mode.
     * True: wait the indexes are ready.
     * False: return immediately no matter the indexes are ready or not.
     */
    bool
    Sync() const;

    /**
     * @brief Set sync mode.
     * True: wait the indexes are ready.
     * False: return immediately no matter the indexes are ready or not.
     */
    void
    SetSync(bool sync);

    /**
     * @brief Set sync mode.
     * True: wait the indexes are ready.
     * False: return immediately no matter the indexes are ready or not.
     */
    CreateIndexRequest&
    WithSync(bool sync);

    /**
     * @brief Timeout in milliseconds.
     */
    int64_t
    TimeoutMs() const;

    /**
     * @brief Set timeout in milliseconds.
     */
    void
    SetTimeoutMs(int64_t timeout_ms);

    /**
     * @brief Set timeout in milliseconds.
     */
    CreateIndexRequest&
    WithTimeoutMs(int64_t timeout_ms);

 private:
    std::vector<IndexDesc> indexes_;
    bool sync_{true};
    int64_t timeout_ms_{60000};
};

}  // namespace milvus
