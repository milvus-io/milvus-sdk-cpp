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

#include "types/DataType.h"

namespace milvus {
namespace param {
/**
 * @brief Field data for insert
 */
template <typename T>
class Field {
    /**
     * @brief Field name
     */
    std::string name_{};

    /**
     * @brief Inserted data type
     */
    DataType type_{DataType::UNKNOWN};

    /**
     * @brief Inserted data values
     */
    std::vector<T> values_;

 public:
    /**
     * @brief constructor for field, data type will set by T
     *
     * @param [in] name name of the field
     * @param [in] values data values
     *
     * @note supported T: bool, int8_t, int16_t, int32_t, int64_t, float, double,
     *                    std::string, std::vector<float>, std::vector<char>
     */
    Field(std::string name, std::vector<T> values);

    const std::string&
    Name() const {
        return name_;
    }

    DataType
    Type() const {
        return type_;
    }

    const std::vector<T>&
    Values() const {
        return values_;
    }
};
}  // namespace param
}  // namespace milvus
