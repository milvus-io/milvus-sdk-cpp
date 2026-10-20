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

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::DeleteRLSPrincipalTags().
 */
class MILVUS_SDK_API DeleteRLSPrincipalTagsRequest {
 public:
    /**
     * @brief Constructor
     */
    DeleteRLSPrincipalTagsRequest() = default;

    /**
     * @brief Get database name.
     * @return the database name.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name.
     * @param [in] db_name the database name.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set database name.
     * @param [in] db_name the database name.
     */
    DeleteRLSPrincipalTagsRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Get collection name.
     * @return the collection name.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set collection name.
     * @param [in] collection_name the collection name.
     */
    void
    SetCollectionName(const std::string& collection_name);

    /**
     * @brief Set collection name.
     * @param [in] collection_name the collection name.
     */
    DeleteRLSPrincipalTagsRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Get the application-level principal name.
     * @return the principal name.
     */
    const std::string&
    PrincipalName() const;

    /**
     * @brief Set the application-level principal name.
     * @param [in] principal_name the principal name.
     */
    void
    SetPrincipalName(const std::string& principal_name);

    /**
     * @brief Set the application-level principal name.
     * @param [in] principal_name the principal name.
     */
    DeleteRLSPrincipalTagsRequest&
    WithPrincipalName(const std::string& principal_name);

    /**
     * @brief Get the tag keys to delete; empty means all tags of the principal.
     * @return the tag keys.
     */
    const std::vector<std::string>&
    TagKeys() const;

    /**
     * @brief Set the tag keys to delete; empty means all tags of the principal.
     * @param [in] tag_keys the tag keys.
     */
    void
    SetTagKeys(std::vector<std::string>&& tag_keys);

    /**
     * @brief Set the tag keys to delete; empty means all tags of the principal.
     * @param [in] tag_keys the tag keys.
     */
    DeleteRLSPrincipalTagsRequest&
    WithTagKeys(std::vector<std::string>&& tag_keys);

 private:
    std::string db_name_;
    std::string collection_name_;
    std::string principal_name_;
    std::vector<std::string> tag_keys_;
};

}  // namespace milvus
