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

#include "./CollectionRequestBase.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::RenameCollection()
 */
class MILVUS_SDK_API RenameCollectionRequest : public CollectionRequestBase<RenameCollectionRequest> {
 public:
    /**
     * @brief Constructor
     */
    RenameCollectionRequest() = default;

    /**
     * @brief New name of the collection.
     * @return the new collection name.
     */
    const std::string&
    NewCollectionName() const;

    /**
     * @brief Set new name of the collection.
     * @param [in] collection_name the collection name.
     */
    void
    SetNewCollectionName(const std::string& collection_name);

    /**
     * @brief Set new name of the collection.
     * @param [in] collection_name the collection name.
     */
    RenameCollectionRequest&
    WithNewCollectionName(const std::string& collection_name);

    /**
     * @brief Target database name. An empty value renames the collection within the source database.
     * @return the target database name.
     */
    const std::string&
    TargetDatabaseName() const;

    /**
     * @brief Set the target database name.
     * @param [in] db_name the DB name.
     */
    void
    SetTargetDatabaseName(const std::string& db_name);

    /**
     * @brief Set the target database name.
     * @param [in] db_name the DB name.
     */
    RenameCollectionRequest&
    WithTargetDatabaseName(const std::string& db_name);

 private:
    std::string new_collection_name_;
    std::string target_db_name_;
};

}  // namespace milvus
