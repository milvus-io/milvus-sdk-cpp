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

#include <unordered_map>

#include "../../types/ConsistencyLevel.h"
#include "../../types/DataType.h"
#include "../../types/MetricType.h"
#include "./CollectionRequestBase.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::CreateCollection() to create a simple collection.
 * The simple collection has only two fields: primary field and vector field.
 * The primary field can be INT64 or VARCHAR type.
 * The vector field is FLOAT_VECTOR type, index is AUTOINDEX.
 */
class CreateSimpleCollectionRequest : public CollectionRequestBase<CreateSimpleCollectionRequest> {
 public:
    /**
     * @brief Constructor
     */
    CreateSimpleCollectionRequest() = default;

    /**
     * @brief Name of the primary field.
     */
    const std::string&
    PrimaryFieldName() const;

    /**
     * @brief Set name of the primary field.
     * Default value is "id".
     */
    void
    SetPrimaryFieldName(const std::string& primary_field_name);

    /**
     * @brief Set name of the primary field.
     * Default value is "id".
     */
    CreateSimpleCollectionRequest&
    WithPrimaryFieldName(const std::string& primary_field_name);

    /**
     * @brief Data type of the primary field.
     */
    DataType
    PrimaryFieldType() const;

    /**
     * @brief Set data type of the primary field.
     * Default value is INT64.
     */
    void
    SetPrimaryFieldType(DataType primary_field_type);

    /**
     * @brief Set data type of the primary field.
     * Default value is INT64.
     */
    CreateSimpleCollectionRequest&
    WithPrimaryFieldType(DataType primary_field_type);

    /**
     * @brief Name of the vector field.
     */
    const std::string&
    VectorFieldName() const;

    /**
     * @brief Set name of the vector field.
     * Default value is "vector".
     */
    void
    SetVectorFieldName(const std::string& vector_field_name);

    /**
     * @brief Set name of the vector field.
     * Default value is "vector".
     */
    CreateSimpleCollectionRequest&
    WithVectorFieldName(const std::string& vector_field_name);

    /**
     * @brief Dimension of the vector field.
     */
    int64_t
    Dimension() const;

    /**
     * @brief Set dimension of the vector field.
     * Default value is 0. User must specify a non-zero value for dimension.
     */
    void
    SetDimension(int64_t dimension);

    /**
     * @brief Set dimension of the vector field.
     * Default value is 0. User must specify a non-zero value for dimension.
     */
    CreateSimpleCollectionRequest&
    WithDimension(int64_t dimension);

    /**
     * @brief Consistency level of the collection.
     */
    milvus::ConsistencyLevel
    ConsistencyLevel() const;

    /**
     * @brief Set consistency level of the collection.
     * Default value is BOUNDED.
     */
    void
    SetConsistencyLevel(milvus::ConsistencyLevel level);

    /**
     * @brief Set consistency level of the collection.
     * Default value is BOUNDED.
     */
    CreateSimpleCollectionRequest&
    WithConsistencyLevel(milvus::ConsistencyLevel level);

    /**
     * @brief Metric type of the collection.
     */
    milvus::MetricType
    MetricType() const;

    /**
     * @brief Set metric type of the collection.
     * Default value is COSINE.
     */
    void
    SetMetricType(milvus::MetricType metric_type);

    /**
     * @brief Set metric type of the collection.
     * Default value is COSINE.
     */
    CreateSimpleCollectionRequest&
    WithMetricType(milvus::MetricType metric_type);

    /**
     * @brief Auto ID generation flag.
     */
    bool
    AutoID() const;

    /**
     * @brief Set auto ID generation flag.
     * Default value is false.
     */
    void
    SetAutoID(bool auto_id);

    /**
     * @brief Set auto ID generation flag.
     * Default value is false.
     */
    CreateSimpleCollectionRequest&
    WithAutoID(bool auto_id);

    /**
     * @brief Dynamic field enable flag.
     */
    bool
    EnableDynamicField() const;

    /**
     * @brief Set dynamic field enable flag.
     * Default value is true.
     */
    void
    SetEnableDynamicField(bool enable_dynamic_field);

    /**
     * @brief Set dynamic field enable flag.
     * Default value is true.
     */
    CreateSimpleCollectionRequest&
    WithEnableDynamicField(bool enable_dynamic_field);

    /**
     * @brief Maximum length of the primary field if it is a VARCHAR.
     */
    int64_t
    MaxLength() const;

    /**
     * @brief Set maximum length of the primary field if it is a VARCHAR.
     * Default value is 65535.
     */
    void
    SetMaxLength(int64_t max_length);

    /**
     * @brief Set maximum length of the primary field if it is a VARCHAR.
     * Default value is 65535.
     */
    CreateSimpleCollectionRequest&
    WithMaxLength(int64_t max_length);

 private:
    std::string primary_field_name_{"id"};
    DataType primary_field_type_{DataType::INT64};
    std::string vector_field_name_{"vector"};
    int64_t dimension_{0};  // require user to specify dimension
    milvus::ConsistencyLevel level_{milvus::ConsistencyLevel::BOUNDED};
    milvus::MetricType metric_type_{milvus::MetricType::COSINE};
    bool auto_id_{false};
    bool enable_dynamic_field_{true};
    int64_t max_length_{65535};  // if primary field is varchar
};

}  // namespace milvus
