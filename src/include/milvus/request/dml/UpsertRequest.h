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
#include "milvus/Export.h"
#include "milvus/types/FieldPartialUpdateOp.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::Upsert()
 *
 * Override mode inserts a new entity or replaces an existing one by primary key. Merge mode
 * (WithPartialUpdate(true)) updates only the supplied fields of an existing entity.
 * @par Example
 * @code
 * milvus::UpsertResponse response;
 * milvus::UpsertRequest request;
 * request.WithCollectionName("demo")
 *     .AddRowData({{ "id", 1 }, { "vector", std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f} }});
 * auto status = client->Upsert(request, response);
 * @endcode
 */
class MILVUS_SDK_API UpsertRequest : public InsertRequest {
 public:
    /**
     * @brief Constructor
     */
    UpsertRequest() = default;

    /**
     * @brief Set database name.
     * If database name is empty, will list collections of the default database.
     * @param [in] db_name the DB name.
     */
    UpsertRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Set name of the collection.
     * @param [in] collection_name the collection name.
     */
    UpsertRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Set new name of the partition.
     * If partition name is empty, it will insert data into the default partition.
     * @param [in] partition_name the partition name.
     */
    UpsertRequest&
    WithPartitionName(const std::string& partition_name);

    /**
     * @brief Set fields data with fluent interface.
     * ColumnsData and RowsData cannot both be set.
     * @param [in] columns_data the columns data.
     */
    UpsertRequest&
    WithColumnsData(std::vector<FieldDataPtr>&& columns_data);

    /**
     * @brief Set a field data with fluent interface.
     * ColumnsData and RowsData cannot both be set.
     * @param [in] column_data the column data.
     */
    UpsertRequest&
    AddColumnData(const FieldDataPtr& column_data);

    /**
     * @brief Set entity rows with fluent interface.
     * ColumnsData and RowsData cannot both be set.
     * @param [in] rows_data the rows data.
     */
    UpsertRequest&
    WithRowsData(EntityRows&& rows_data);

    /**
     * @brief Add an entity row with the fluent interface.
     * ColumnsData and RowsData cannot both be set.
     * @param [in] row_data the row data.
     */
    UpsertRequest&
    AddRowData(EntityRow&& row_data);

    /**
     * @brief Get partial update or not.
     * @return the partial update.
     */
    bool
    PartialUpdate() const;

    /**
     * @brief Set partial update.
     * If True, only the specified fields will be updated while others remain unchanged.
     * Default is False.
     * @param [in] partial_update the partial update.
     */
    void
    SetPartialUpdate(bool partial_update);

    /**
     * @brief Set database name.
     * If True, only the specified fields will be updated while others remain unchanged.
     * Default is False.
     * @param [in] partial_update the partial update.
     */
    UpsertRequest&
    WithPartialUpdate(bool partial_update);

    /**
     * @brief Get per-field partial update operations.
     * @return the field ops.
     */
    const std::vector<FieldPartialUpdateOp>&
    FieldOps() const;

    /**
     * @brief Set per-field partial update operations.
     * ARRAY_APPEND and ARRAY_REMOVE automatically enable partial update semantics.
     * @param [in] field_ops the field ops.
     */
    void
    SetFieldOps(std::vector<FieldPartialUpdateOp>&& field_ops);

    /**
     * @brief Set per-field partial update operations with fluent interface.
     * @param [in] field_ops the field ops.
     */
    UpsertRequest&
    WithFieldOps(std::vector<FieldPartialUpdateOp>&& field_ops);

    /**
     * @brief Add a per-field partial update operation.
     * @param [in] field_op the field op.
     */
    UpsertRequest&
    AddFieldOp(FieldPartialUpdateOp field_op);

 private:
    bool partial_update_{false};
    std::vector<FieldPartialUpdateOp> field_ops_;
};

}  // namespace milvus
