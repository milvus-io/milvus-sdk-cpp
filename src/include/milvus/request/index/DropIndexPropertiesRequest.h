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

#include "./IndexRequestBase.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::DropIndexProperties()
 */
class MILVUS_SDK_API DropIndexPropertiesRequest : public IndexRequestBase<DropIndexPropertiesRequest> {
 public:
    /**
     * @brief Constructor
     */
    DropIndexPropertiesRequest() = default;

    /**
     * @brief Name of the index.
     * @return the index name.
     */
    const std::string&
    IndexName() const;

    /**
     * @brief Set name of the index.
     * Currently, this API only supports index_name.
     * @param [in] index_name the index name.
     */
    void
    SetIndexName(const std::string& index_name);

    /**
     * @brief Set name of the index.
     * Currently, this API only supports index_name.
     * @param [in] index_name the index name.
     */
    DropIndexPropertiesRequest&
    WithIndexName(const std::string& index_name);

    /**
     * @brief Get deleted keys.
     * @return the property keys.
     */
    const std::set<std::string>&
    PropertyKeys() const;

    /**
     * @brief Set deleted keys of this index.
     * @param [in] keys the keys.
     */
    void
    SetPropertyKeys(std::set<std::string>&& keys);

    /**
     * @brief Set deleted keys of this index.
     * @param [in] keys the keys.
     */
    DropIndexPropertiesRequest&
    WithPropertyKeys(std::set<std::string>&& keys);

    /**
     * @brief Set a key to be deleted.
     * @param [in] key the key.
     */
    DropIndexPropertiesRequest&
    AddPropertyKey(const std::string& key);

 private:
    std::string index_name_;
    std::set<std::string> property_keys_;
};

}  // namespace milvus
