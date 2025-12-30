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

#include "AliasDesc.h"

namespace milvus {

/**
 * @brief Aluas description. Used by MilvusClient::DescribeAlias()
 */
class AliasDesc {
 public:
    /**
     * @brief Construct a new Desc object.
     */
    AliasDesc();

    /**
     * @brief Construct a new Desc object.
     *
     * @param alias_name  alias name
     * @param db_name database name
     * @param collection_name collection name
     */
    AliasDesc(std::string alias_name, std::string db_name, std::string collection_name);

    /**
     * @brief Alias name.
     */
    const std::string&
    Name() const;

    /**
     * @brief Set name of the alias.
     */
    void
    SetName(const std::string& name);

    /**
     * @brief Database name which the alias belong to.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name which the alias belong to.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Collection name.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set collection name.
     */
    void
    SetCollectionName(const std::string& collection_name);

 private:
    std::string name_;
    std::string db_name_;
    std::string collection_name_;
};

}  // namespace milvus
