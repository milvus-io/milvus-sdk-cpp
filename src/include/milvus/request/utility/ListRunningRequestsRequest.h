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

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::ListRunningRequests()
 */
class MILVUS_SDK_API ListRunningRequestsRequest {
 public:
    /**
     * @brief Constructor
     */
    ListRunningRequestsRequest() = default;

    /**
     * @brief Database name to match exactly; empty means any database.
     * @return the database name.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name to match exactly; empty means any database.
     * @param [in] db_name the DB name.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set database name to match exactly; empty means any database.
     * @param [in] db_name the DB name.
     */
    ListRunningRequestsRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Collection name to match exactly; empty means any collection.
     * @return the collection name.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set collection name to match exactly; empty means any collection.
     * @param [in] collection_name the collection name.
     */
    void
    SetCollectionName(const std::string& collection_name);

    /**
     * @brief Set collection name to match exactly; empty means any collection.
     * @param [in] collection_name the collection name.
     */
    ListRunningRequestsRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief User whose requests to list; empty means every user. A caller without the
     * ListRunningRequests privilege may only see their own requests: empty then means the
     * caller, and any other name is denied.
     * @return the user.
     */
    const std::string&
    User() const;

    /**
     * @brief Set the user whose requests to list.
     * @param [in] user the user.
     */
    void
    SetUser(const std::string& user);

    /**
     * @brief Set the user whose requests to list.
     * @param [in] user the user.
     */
    ListRunningRequestsRequest&
    WithUser(const std::string& user);

    /**
     * @brief Minimum time, in milliseconds, a request has been running; 0 means no constraint.
     * @return the min elapsed time.
     */
    int64_t
    MinElapsedMs() const;

    /**
     * @brief Set the minimum time, in milliseconds, a request has been running.
     * @param [in] min_elapsed_ms the min elapsed time.
     */
    void
    SetMinElapsedMs(int64_t min_elapsed_ms);

    /**
     * @brief Set the minimum time, in milliseconds, a request has been running.
     * @param [in] min_elapsed_ms the min elapsed time.
     */
    ListRunningRequestsRequest&
    WithMinElapsedMs(int64_t min_elapsed_ms);

 private:
    std::string db_name_;
    std::string collection_name_;
    std::string user_;
    int64_t min_elapsed_ms_ = 0;
};

}  // namespace milvus
