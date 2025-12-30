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

#include <unordered_map>

#include "./IndexRequestBase.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::AlterIndexProperties()
 */
class AlterIndexPropertiesRequest : public IndexRequestBase {
 public:
    /**
     * @brief Constructor
     */
    AlterIndexPropertiesRequest() = default;

    /**
     * @brief Set database name in which the collection is created.
     */
    AlterIndexPropertiesRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Set name of the collection.
     */
    AlterIndexPropertiesRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Name of the index.
     */
    const std::string&
    IndexName() const;

    /**
     * @brief Set name of the index.
     * Currently this api only supports index_name.
     */
    void
    SetIndexName(const std::string& index_name);

    /**
     * @brief Set name of the index.
     * Currently this api only supports index_name.
     */
    AlterIndexPropertiesRequest&
    WithIndexName(const std::string& index_name);

    /**
     * @brief Get altered properties.
     */
    const std::unordered_map<std::string, std::string>&
    Properties() const;

    /**
     * @brief Set altered properties of this index.
     */
    void
    SetProperties(std::unordered_map<std::string, std::string>&& properties);

    /**
     * @brief Set altered properties of this index.
     */
    AlterIndexPropertiesRequest&
    WithProperties(std::unordered_map<std::string, std::string>&& properties);

    /**
     * @brief Add a property of this index.
     */
    AlterIndexPropertiesRequest&
    AddProperty(const std::string& key, const std::string& property);

 private:
    std::string index_name_;
    std::unordered_map<std::string, std::string> properties_;
};

}  // namespace milvus
