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

namespace milvus {

/**
 * @brief Used by MilvusClientV2::ListAliases()
 */
class ListAliasesResponse {
 public:
    /**
     * @brief Constructor
     */
    ListAliasesResponse() = default;

    // Getter and Setter for db_name_
    const std::string&
    DatabaseName() const;
    void
    SetDatabaseName(const std::string& db_name);

    // Getter and Setter for collection_name_
    const std::string&
    CollectionName() const;
    void
    SetCollectionName(const std::string& collection_name);

    // Getter and Setter for aliases_
    const std::vector<std::string>&
    Aliases() const;
    void
    SetAliases(std::vector<std::string>&& aliases);

 private:
    std::string db_name_;
    std::string collection_name_;
    std::vector<std::string> aliases_;
};

}  // namespace milvus
