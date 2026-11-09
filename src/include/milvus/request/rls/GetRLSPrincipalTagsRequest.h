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

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::GetRLSPrincipalTags().
 */
class MILVUS_SDK_API GetRLSPrincipalTagsRequest {
 public:
    /**
     * @brief Constructor
     */
    GetRLSPrincipalTagsRequest() = default;

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
    GetRLSPrincipalTagsRequest&
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
    GetRLSPrincipalTagsRequest&
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
    GetRLSPrincipalTagsRequest&
    WithPrincipalName(const std::string& principal_name);

 protected:
    std::string db_name_;
    std::string collection_name_;
    std::string principal_name_;
};

}  // namespace milvus
