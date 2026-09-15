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

#include "../../types/FieldSchema.h"
#include "../../types/Function.h"
#include "../../types/IndexDesc.h"
#include "./CollectionRequestBase.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::AddFunctionField()
 */
class MILVUS_SDK_API AddFunctionFieldRequest : public CollectionRequestBase<AddFunctionFieldRequest> {
 public:
    /**
     * @brief Constructor
     */
    AddFunctionFieldRequest() = default;

    /**
     * @brief Get the function output field schema.
     *
     * BM25 requires SPARSE_FLOAT_VECTOR and MinHash requires BINARY_VECTOR.
     * @return the field.
     */
    const FieldSchema&
    Field() const;

    /**
     * @brief Set the function output field schema.
     *
     * BM25 requires SPARSE_FLOAT_VECTOR and MinHash requires BINARY_VECTOR.
     * @param [in] field_schema the field schema.
     */
    void
    SetField(FieldSchema&& field_schema);

    /**
     * @brief Set the function output field schema.
     *
     * BM25 requires SPARSE_FLOAT_VECTOR and MinHash requires BINARY_VECTOR.
     * @param [in] field_schema the field schema.
     */
    AddFunctionFieldRequest&
    WithField(FieldSchema&& field_schema);

    /**
     * @brief Get the function to be added.
     *
     * AddFunctionField currently supports BM25 and MinHash functions.
     * @return the function.
     */
    const FunctionPtr&
    Function() const;

    /**
     * @brief Set the function to be added.
     *
     * AddFunctionField currently supports BM25 and MinHash functions.
     */
    void
    SetFunction(const FunctionPtr& function);

    /**
     * @brief Set the function to be added.
     *
     * AddFunctionField currently supports BM25 and MinHash functions.
     */
    AddFunctionFieldRequest&
    WithFunction(const FunctionPtr& function);

    /**
     * @brief Get the index bound to the function output field.
     *
     * The bound index is required and must use an explicit index type.
     * @return the index.
     */
    const IndexDesc&
    Index() const;

    /**
     * @brief Set the index bound to the function output field.
     *
     * The bound index is required and must use an explicit index type.
     * @param [in] index the index.
     */
    void
    SetIndex(IndexDesc&& index);

    /**
     * @brief Set the index bound to the function output field.
     *
     * The bound index is required and must use an explicit index type.
     * @param [in] index the index.
     */
    AddFunctionFieldRequest&
    WithIndex(IndexDesc&& index);

 private:
    FieldSchema field_;
    FunctionPtr function_;
    IndexDesc index_;
};

}  // namespace milvus
