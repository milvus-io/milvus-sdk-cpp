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

#include <set>
#include <string>

namespace milvus {

/**
 * @brief Used by MilvusClientV2::Flush()
 */
class FlushRequest {
 public:
    /**
     * @brief Constructor
     */
    FlushRequest() = default;

    /**
     * @brief Database name in which the collections are created.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name in which the collections are created.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set database name in which the collections are created.
     */
    FlushRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Get collection names to be flushed.
     */
    const std::set<std::string>&
    CollectionNames() const;

    /**
     * @brief Set collection names to be flushed.
     */
    void
    SetCollectionNames(std::set<std::string>&& names);

    /**
     * @brief Set collection names to be flushed.
     */
    FlushRequest&
    WithCollectionNames(std::set<std::string>&& names);

    /**
     * @brief Add a collection name to be flushed.
     */
    FlushRequest&
    AddCollectionName(const std::string& name);

    /**
     * @brief Get milliseconds to wait the flush action done.
     */
    int64_t
    WaitFlushedMs() const;

    /**
     * @brief Set milliseconds to wait the flush action done.
     * If the WaitFlushedMs is larger than zero, the Flush() will call GetFlushState() to check related segments state,
     * to make sure the buffer persisted successfully.
     */
    void
    SetWaitFlushedMs(int64_t ms);

    /**
     * @brief Set milliseconds to wait the flush action done.
     * If the WaitFlushedMs is larger than zero, the Flush() will call GetFlushState() to check related segments state,
     * to make sure the buffer persisted successfully.
     */
    FlushRequest&
    WithWaitFlushedMs(int64_t ms);

 private:
    std::string db_name_;
    std::set<std::string> collection_names_;
    int64_t wait_flushed_ms_;
};

}  // namespace milvus
