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

#include "SearchAggregation.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Describes a scalar field and direction used to order query or search results.
 */
class MILVUS_SDK_API OrderByField {
 public:
    /**
     * @brief Constructor
     */
    OrderByField() = default;
    explicit OrderByField(std::string field_name, AggregationDirection direction = AggregationDirection::ASC);

    /**
     * @brief Get the field name.
     * @return the field name.
     */
    const std::string&
    FieldName() const;

    /**
     * @brief Set the field name.
     *
     * @param [in] field_name
     */
    void
    SetFieldName(std::string field_name);

    /**
     * @brief Set the field name.
     *
     * @param [in] field_name
     */
    OrderByField&
    WithFieldName(std::string field_name);

    /**
     * @brief Get the sort direction.
     * @return the direction.
     */
    AggregationDirection
    Direction() const;

    /**
     * @brief Set the sort direction.
     *
     * @param [in] direction
     */
    void
    SetDirection(AggregationDirection direction);

    /**
     * @brief Set the sort direction.
     *
     * @param [in] direction
     */
    OrderByField&
    WithDirection(AggregationDirection direction);

 private:
    std::string field_name_;
    AggregationDirection direction_{AggregationDirection::ASC};
};

}  // namespace milvus
