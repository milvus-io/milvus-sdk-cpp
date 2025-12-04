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

namespace milvus {

/**
 * @brief Used by MilvusClientV2::ListCollections()
 */
class ListCollectionsRequest {
 public:
    /**
     * @brief Constructor
     */
    ListCollectionsRequest() = default;

    /**
     * @brief Database name.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name.
     * If database name is empty, will list collections of the default database
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set database name.
     * If database name is empty, will list collections of the default database
     */
    ListCollectionsRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Only show loaded collections or show all collections.
     */
    bool
    OnlyShowLoaded() const;

    /**
     * @brief Set the flag only show loaded collections or show all collections.
     */
    void
    SetOnlyShowLoaded(bool only_show_loaded);

    /**
     * @brief Set the flag only show loaded collections or show all collections.
     */
    ListCollectionsRequest&
    WithOnlyShowLoaded(bool only_show_loaded);

 private:
    std::string db_name_;
    bool only_show_loaded_{false};
};

}  // namespace milvus
