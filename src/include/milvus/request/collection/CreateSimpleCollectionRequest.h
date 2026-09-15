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
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::CreateCollection() to create a simple collection.
 * The simple collection has only two fields: primary field and vector field.
 * The primary field can be INT64 or VARCHAR type.
 * The vector field is FLOAT_VECTOR type, index is AUTOINDEX.
 * @par Example
 * @code
 * auto status = client->CreateCollection(milvus::CreateSimpleCollectionRequest()
 *                                            .WithCollectionName("demo")
 *                                            .WithDimension(128)
 *                                            .WithMetricType(milvus::MetricType::L2));
 * @endcode
 */
class MILVUS_SDK_API CreateSimpleCollectionRequest : public CollectionRequestBase<CreateSimpleCollectionRequest> {
 public:
    /**
     * @brief Constructor
     */
    CreateSimpleCollectionRequest() = default;

    /**
     * @brief Name of the primary field.
     * @return the primary field name.
     */
    const std::string&
    PrimaryFieldName() const;

    /**
     * @brief Set name of the primary field.
     * Default value is "id".
     * @param [in] primary_field_name the primary field name.
     */
    void
    SetPrimaryFieldName(const std::string& primary_field_name);

    /**
     * @brief Set name of the primary field.
     * Default value is "id".
     * @param [in] primary_field_name the primary field name.
     */
    CreateSimpleCollectionRequest&
    WithPrimaryFieldName(const std::string& primary_field_name);

    /**
     * @brief Data type of the primary field.
     * @return the primary field type.
     */
    DataType
    PrimaryFieldType() const;

    /**
     * @brief Set data type of the primary field.
     * Default value is INT64.
     * @param [in] primary_field_type the primary field type.
     */
    void
    SetPrimaryFieldType(DataType primary_field_type);

    /**
     * @brief Set data type of the primary field.
     * Default value is INT64.
     * @param [in] primary_field_type the primary field type.
     */
    CreateSimpleCollectionRequest&
    WithPrimaryFieldType(DataType primary_field_type);

    /**
     * @brief Name of the vector field.
     * @return the vector field name.
     */
    const std::string&
    VectorFieldName() const;

    /**
     * @brief Set name of the vector field.
     * Default value is "vector".
     * @param [in] vector_field_name the vector field name.
     */
    void
    SetVectorFieldName(const std::string& vector_field_name);

    /**
     * @brief Set name of the vector field.
     * Default value is "vector".
     * @param [in] vector_field_name the vector field name.
     */
    CreateSimpleCollectionRequest&
    WithVectorFieldName(const std::string& vector_field_name);

    /**
     * @brief Dimension of the vector field.
     * @return the dimension.
     */
    int64_t
    Dimension() const;

    /**
     * @brief Set dimension of the vector field.
     * Default value is 0. User must specify a non-zero value for dimension.
     * @param [in] dimension the dimension.
     */
    void
    SetDimension(int64_t dimension);

    /**
     * @brief Set dimension of the vector field.
     * Default value is 0. User must specify a non-zero value for dimension.
     * @param [in] dimension the dimension.
     */
    CreateSimpleCollectionRequest&
    WithDimension(int64_t dimension);

    /**
     * @brief Consistency level of the collection.
     * @return the consistency level.
     */
    milvus::ConsistencyLevel
    ConsistencyLevel() const;

    /**
     * @brief Set consistency level of the collection.
     * Default value is BOUNDED.
     * @param [in] level the level.
     */
    void
    SetConsistencyLevel(milvus::ConsistencyLevel level);

    /**
     * @brief Set consistency level of the collection.
     * Default value is BOUNDED.
     * @param [in] level the level.
     */
    CreateSimpleCollectionRequest&
    WithConsistencyLevel(milvus::ConsistencyLevel level);

    /**
     * @brief Metric type of the collection.
     * @return the metric type.
     */
    milvus::MetricType
    MetricType() const;

    /**
     * @brief Set metric type of the collection.
     * Default value is COSINE.
     * @param [in] metric_type the metric type.
     */
    void
    SetMetricType(milvus::MetricType metric_type);

    /**
     * @brief Set metric type of the collection.
     * Default value is COSINE.
     * @param [in] metric_type the metric type.
     */
    CreateSimpleCollectionRequest&
    WithMetricType(milvus::MetricType metric_type);

    /**
     * @brief Auto ID generation flag.
     * @return the auto ID.
     */
    bool
    AutoID() const;

    /**
     * @brief Set auto ID generation flag.
     * Default value is false.
     * @param [in] auto_id the auto ID.
     */
    void
    SetAutoID(bool auto_id);

    /**
     * @brief Set auto ID generation flag.
     * Default value is false.
     * @param [in] auto_id the auto ID.
     */
    CreateSimpleCollectionRequest&
    WithAutoID(bool auto_id);

    /**
     * @brief Dynamic field enable flag.
     * @return the enable dynamic field.
     */
    bool
    EnableDynamicField() const;

    /**
     * @brief Set dynamic field enable flag.
     * Default value is true.
     * @param [in] enable_dynamic_field the enable dynamic field.
     */
    void
    SetEnableDynamicField(bool enable_dynamic_field);

    /**
     * @brief Set dynamic field enable flag.
     * Default value is true.
     * @param [in] enable_dynamic_field the enable dynamic field.
     */
    CreateSimpleCollectionRequest&
    WithEnableDynamicField(bool enable_dynamic_field);

    /**
     * @brief Maximum length of the primary field if it is a VARCHAR.
     * @return the max length.
     */
    int64_t
    MaxLength() const;

    /**
     * @brief Set maximum length of the primary field if it is a VARCHAR.
     * Default value is 65535.
     * @param [in] max_length the max length.
     */
    void
    SetMaxLength(int64_t max_length);

    /**
     * @brief Set maximum length of the primary field if it is a VARCHAR.
     * Default value is 65535.
     * @param [in] max_length the max length.
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
