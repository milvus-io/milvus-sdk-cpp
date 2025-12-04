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
#include <unordered_map>

namespace milvus {

/**
 * @brief Used by MilvusClientV2::CreateDatabase()
 */
class CreateDatabaseRequest {
 public:
    /**
     * @brief Constructor
     */
    CreateDatabaseRequest() = default;

    /**
     * @brief Database name.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set database name.
     */
    CreateDatabaseRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Get properties of this database.
     */
    const std::unordered_map<std::string, std::string>&
    Properties() const;

    /**
     * @brief Set properties of this database.
     */
    void
    SetProperties(std::unordered_map<std::string, std::string>&& properties);

    /**
     * @brief Set properties of this database.
     */
    CreateDatabaseRequest&
    WithProperties(std::unordered_map<std::string, std::string>&& properties);

    /**
     * @brief Add a property of this database.
     */
    CreateDatabaseRequest&
    AddProperty(const std::string& key, const std::string& property);

 private:
    std::string db_name_;
    std::unordered_map<std::string, std::string> properties_;
};

}  // namespace milvus
