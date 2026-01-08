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

#include "./DBRequestBase.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::DropDatabaseProperties()
 */
class DropDatabasePropertiesRequest : public DBRequestBase<DropDatabasePropertiesRequest> {
 public:
    /**
     * @brief Constructor
     */
    DropDatabasePropertiesRequest() = default;

    /**
     * @brief Get deleted keys.
     */
    const std::set<std::string>&
    PropertyKeys() const;

    /**
     * @brief Set deleted keys of this database.
     */
    void
    SetPropertyKeys(std::set<std::string>&& keys);

    /**
     * @brief Set deleted keys of this database.
     */
    DropDatabasePropertiesRequest&
    WithPropertyKeys(std::set<std::string>&& keys);

    /**
     * @brief Add a key to be deleted.
     */
    DropDatabasePropertiesRequest&
    AddPropertyKey(const std::string& key);

 private:
    std::set<std::string> property_keys_;
};

}  // namespace milvus
