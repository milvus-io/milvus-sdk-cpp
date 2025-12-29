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

#include "./InsertRequest.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::Upsert()
 */
class UpsertRequest : public InsertRequest {
 public:
    /**
     * @brief Constructor
     */
    UpsertRequest() = default;

    /**
     * @brief Set database name.
     * If database name is empty, will list collections of the default database.
     */
    UpsertRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Set name of the collection.
     */
    UpsertRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Set new name of the partition.
     * If partition name is empty, it will insert data into the default partition.
     */
    UpsertRequest&
    WithPartitionName(const std::string& partition_name);

    /**
     * @brief Set fields data with fluent interface.
     * Not allow to set ColumnsData and RowsData both.
     */
    UpsertRequest&
    WithColumnsData(std::vector<FieldDataPtr>&& columns_data);

    /**
     * @brief Set a field data with fluent interface.
     * Not allow to set ColumnsData and RowsData both.
     */
    UpsertRequest&
    AddColumnData(const FieldDataPtr& column_data);

    /**
     * @brief Set entity rows with fluent interface.
     * Not allow to set ColumnsData and RowsData both.
     */
    UpsertRequest&
    WithRowsData(EntityRows&& rows_data);

    /**
     * @brief Add en entity rows with fluent interface.
     * Not allow to set ColumnsData and RowsData both.
     */
    UpsertRequest&
    AddRowData(EntityRow&& row_data);

    /**
     * @brief Get partial update or not.
     */
    bool
    PartialUpdate() const;

    /**
     * @brief Set partial update.
     * If True, only the specified fields will be updated while others remain unchanged.
     * Default is False.
     */
    void
    SetPartialUpdate(bool partial_update);

    /**
     * @brief Set database name.
     * If True, only the specified fields will be updated while others remain unchanged.
     * Default is False.
     */
    UpsertRequest&
    WithPartialUpdate(bool partial_update);

 private:
    bool partial_update_{false};
};

}  // namespace milvus
