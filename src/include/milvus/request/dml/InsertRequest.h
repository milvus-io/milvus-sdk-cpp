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
#include "./DMLRequestBase.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::Insert()
 *
 * Insert does not check duplicate primary keys; use Upsert to update or avoid duplicates.
 * @par Example
 * @code
 * milvus::InsertResponse response;
 * milvus::InsertRequest request;
 * request.WithCollectionName("demo")
 *     .AddRowData({{ "id", 1 }, { "vector", std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f} }})
 *     .AddRowData({{ "id", 2 }, { "vector", std::vector<float>{0.5f, 0.6f, 0.7f, 0.8f} }});
 * auto status = client->Insert(request, response);
 * @endcode
 */
class MILVUS_SDK_API InsertRequest : public DMLRequestBase<InsertRequest> {
 public:
    /**
     * @brief Constructor
     */
    InsertRequest() = default;

    /**
     * @brief Get fields data.
     * @return the columns data.
     */
    const std::vector<FieldDataPtr>&
    ColumnsData() const;

    /**
     * @brief Set fields data.
     * ColumnsData and RowsData cannot both be set.
     * @param [in] columns_data the columns data.
     */
    void
    SetColumnsData(std::vector<FieldDataPtr>&& columns_data);

    /**
     * @brief Set fields data with fluent interface.
     * ColumnsData and RowsData cannot both be set.
     * @param [in] columns_data the columns data.
     */
    InsertRequest&
    WithColumnsData(std::vector<FieldDataPtr>&& columns_data);

    /**
     * @brief Set a field data with fluent interface.
     * ColumnsData and RowsData cannot both be set.
     * @param [in] column_data the column data.
     */
    InsertRequest&
    AddColumnData(const FieldDataPtr& column_data);

    /**
     * @brief Get entity rows.
     * @return the rows data.
     */
    const EntityRows&
    RowsData() const;

    /**
     * @brief Set entity rows.
     * ColumnsData and RowsData cannot both be set.
     * @param [in] rows_data the rows data.
     */
    void
    SetRowsData(EntityRows&& rows_data);

    /**
     * @brief Set entity rows with fluent interface.
     * ColumnsData and RowsData cannot both be set.
     * @param [in] rows_data the rows data.
     */
    InsertRequest&
    WithRowsData(EntityRows&& rows_data);

    /**
     * @brief Add an entity row with the fluent interface.
     * ColumnsData and RowsData cannot both be set.
     * @param [in] row_data the row data.
     */
    InsertRequest&
    AddRowData(EntityRow&& row_data);

 private:
    std::vector<FieldDataPtr> columns_data_;
    EntityRows rows_data_;
};

}  // namespace milvus
