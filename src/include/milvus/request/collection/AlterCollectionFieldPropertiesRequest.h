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

#include "./CollectionRequestBase.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::AlterCollectionFieldProperties()
 */
class AlterCollectionFieldPropertiesRequest : public CollectionRequestBase {
 public:
    /**
     * @brief Constructor
     */
    AlterCollectionFieldPropertiesRequest() = default;

    /**
     * @brief Set database name in which the collection is created.
     */
    AlterCollectionFieldPropertiesRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Set name of the collection.
     */
    AlterCollectionFieldPropertiesRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Name of the field.
     */
    const std::string&
    FieldName() const;

    /**
     * @brief Set name of the field.
     */
    void
    SetFieldName(const std::string& field_name);

    /**
     * @brief Set name of the field.
     */
    AlterCollectionFieldPropertiesRequest&
    WithFieldName(const std::string& field_name);

    /**
     * @brief Get altered properties.
     */
    const std::unordered_map<std::string, std::string>&
    Properties() const;

    /**
     * @brief Set altered properties of this field.
     */
    void
    SetProperties(std::unordered_map<std::string, std::string>&& properties);

    /**
     * @brief Set altered properties of this field.
     */
    AlterCollectionFieldPropertiesRequest&
    WithProperties(std::unordered_map<std::string, std::string>&& properties);

    /**
     * @brief Set a property of this field.
     */
    AlterCollectionFieldPropertiesRequest&
    AddProperty(const std::string& key, const std::string& property);

 private:
    std::string field_name_;
    std::unordered_map<std::string, std::string> properties_;
};

}  // namespace milvus
