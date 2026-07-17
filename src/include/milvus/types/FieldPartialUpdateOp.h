// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
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
 * @brief Describes how a field value is applied during a partial upsert.
 *
 * @note ARRAY_APPEND and ARRAY_REMOVE require Milvus server v2.6.17 or later. Older servers ignore these operations
 * and may apply the supplied array value using REPLACE semantics.
 */
class MILVUS_SDK_API FieldPartialUpdateOp {
 public:
    /**
     * @brief Operation applied to the matching field during a partial upsert.
     *
     * The field named by FieldName() must also be present in the upsert payload.
     */
    enum class OpType {
        /**
         * @brief Overwrite the existing field value.
         *
         * This operation is supported for all field types and is the default when no operation targets a field.
         */
        REPLACE,

        /**
         * @brief Append the supplied elements to an existing array field.
         *
         * This operation is supported only for array fields. The resulting array length must not exceed the field's
         * max_capacity.
         */
        ARRAY_APPEND,

        /**
         * @brief Remove every occurrence of each supplied element from an existing array field.
         *
         * This operation is supported only for array fields. It has no effect when the existing array is empty or no
         * supplied element matches.
         */
        ARRAY_REMOVE,
    };

    FieldPartialUpdateOp() = default;
    explicit FieldPartialUpdateOp(std::string field_name, OpType op_type = OpType::REPLACE);

    const std::string&
    FieldName() const;

    void
    SetFieldName(std::string field_name);

    FieldPartialUpdateOp&
    WithFieldName(std::string field_name);

    OpType
    GetOpType() const;

    void
    SetOpType(OpType op_type);

    FieldPartialUpdateOp&
    WithOpType(OpType op_type);

 private:
    std::string field_name_;
    OpType op_type_{OpType::REPLACE};
};

}  // namespace milvus
