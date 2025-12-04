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

#include "../../types/FieldData.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::Insert()
 */
class InsertRequest {
 public:
    /**
     * @brief Constructor
     */
    InsertRequest() = default;

    /**
     * @brief Database name.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name.
     * If database name is empty, will list collections of the default database.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set database name.
     * If database name is empty, will list collections of the default database.
     */
    InsertRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Name of the collection.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set name of the collection.
     */
    void
    SetCollectionName(const std::string& collection_name);

    /**
     * @brief Set name of the collection.
     */
    InsertRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Name of the partition.
     */
    const std::string&
    PartitionName() const;

    /**
     * @brief Set name of the partition.
     * If partition name is empty, it will insert data into the default partition.
     */
    void
    SetPartitionName(const std::string& partition_name);

    /**
     * @brief Set new name of the partition.
     * If partition name is empty, it will insert data into the default partition.
     */
    InsertRequest&
    WithPartitionName(const std::string& partition_name);

    /**
     * @brief Get fields data.
     */
    const std::vector<FieldDataPtr>&
    ColumnsData() const;

    /**
     * @brief Set fields data.
     * Not allow to set ColumnsData and RowsData both.
     */
    void
    SetColumnsData(std::vector<FieldDataPtr>&& columns_data);

    /**
     * @brief Set fields data with fluent interface.
     * Not allow to set ColumnsData and RowsData both.
     */
    InsertRequest&
    WithColumnsData(std::vector<FieldDataPtr>&& columns_data);

    /**
     * @brief Set a field data with fluent interface.
     * Not allow to set ColumnsData and RowsData both.
     */
    InsertRequest&
    AddColumnData(const FieldDataPtr& column_data);

    /**
     * @brief Get entity rows.
     */
    const EntityRows&
    RowsData() const;

    /**
     * @brief Set entity rows.
     * Not allow to set ColumnsData and RowsData both.
     */
    void
    SetRowsData(EntityRows&& rows_data);

    /**
     * @brief Set entity rows with fluent interface.
     * Not allow to set ColumnsData and RowsData both.
     */
    InsertRequest&
    WithRowsData(EntityRows&& rows_data);

    /**
     * @brief Add en entity rows with fluent interface.
     * Not allow to set ColumnsData and RowsData both.
     */
    InsertRequest&
    AddRowData(EntityRow&& row_data);

 private:
    std::string db_name_;
    std::string collection_name_;
    std::string partition_name_;
    std::vector<FieldDataPtr> columns_data_;
    EntityRows rows_data_;
};

}  // namespace milvus
