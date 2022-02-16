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

#include <memory>
#include <vector>

#include "../Status.h"
#include "DataType.h"

namespace milvus {
class Field {
 public:
    /**
     * @brief Get field name
     */
    const std::string&
    Name() const {
        return name_;
    }

    /**
     * @brief Get field data tpye
     */
    DataType
    Type() const {
        return data_type_;
    }

    /**
     * @brief Total number of field elements
     */
    virtual size_t
    Count() const = 0;

 protected:
    Field(const std::string& name, DataType data_type) : name_(name), data_type_(data_type) {
    }

 private:
    std::string name_;
    DataType data_type_;
};

using FieldDataPtr = std::shared_ptr<Field>;

template <DataType Dt>
struct DataTypeTraits {
    static const bool is_vector = false;
};

template <>
struct DataTypeTraits<DataType::BINARY_VECTOR> {
    static const bool is_vector = true;
};

template <>
struct DataTypeTraits<DataType::FLOAT_VECTOR> {
    static const bool is_vector = true;
};

template <typename T, DataType Dt>
typename std::enable_if<!DataTypeTraits<Dt>::is_vector, StatusCode>::type
AddElement(const T& element, std::vector<T>& array) {
    array.push_back(element);
    return StatusCode::OK;
}

template <typename T, DataType Dt>
typename std::enable_if<DataTypeTraits<Dt>::is_vector, StatusCode>::type
AddElement(const T& element, std::vector<T>& array) {
    if (element.empty()) {
        return StatusCode::VECTOR_IS_EMPTY;
    }

    if (!array.empty() && element.size() != array.at(0).size()) {
        return StatusCode::DIMENSION_NOT_EQUAL;
    }

    array.emplace_back(element);
    return StatusCode::OK;
}

/**
 * @brief Template class represents column-based data of a field. Avaiable inheritance classes: \n
 *  BoolFieldData for boolean scalar field \n
 *  Int8FieldData for 8-bits integer scalar field \n
 *  Int16FieldData for 16-bits integer scalar field \n
 *  Int32FieldData for 32-bits integer scalar field \n
 *  Int64FieldData for 64-bits integer scalar field \n
 *  FloatFieldData for float scalar field \n
 *  DoubleFieldData for double scalar field \n
 *  StringFieldData for string scalar field (not supported in 2.0, reserved) \n
 *  BinaryVecFieldData for float vector scalar field \n
 *  FloatVecFieldData for binary vector scalar field \n
 */
template <typename T, DataType Dt>
class FieldData : public Field {
 public:
    /**
     * @brief Field element type
     */
    using ElementT = T;

    /**
     * @brief Constructor
     */
    FieldData() : Field("", Dt) {
    }

    /**
     * @brief Constructor
     */
    explicit FieldData(const std::string& name) : Field(name, Dt) {
    }

    /**
     * @brief Constructor
     */
    FieldData(const std::string& name, std::vector<T> data) : Field(name, Dt), data_{std::move(data)} {
    }

    /**
     * @brief Add element to field data
     */
    StatusCode
    Add(const T& element) {
        return AddElement<T, Dt>(element, data_);
    }

    /**
     * @brief Total number of field elements
     */
    size_t
    Count() const final {
        return data_.size();
    }

    /**
     * @brief Field elements array
     */
    const std::vector<T>&
    Data() const {
        return data_;
    }

    /**
     * @brief Field elements array
     */
    std::vector<T>&
    Data() {
        return data_;
    }

 private:
    std::vector<T> data_;
};

/**
 * @brief To test two FieldData are equal
 */
template <typename T, DataType Dt>
bool
operator==(const FieldData<T, Dt>& lhs, const FieldData<T, Dt>& rhs) {
    return lhs.Name() == rhs.Name() && lhs.Count() == rhs.Count() && lhs.Data() == rhs.Data();
}

/**
 * @brief To test two FieldData are equal
 */
template <typename T, DataType Dt>
bool
operator==(const FieldData<T, Dt>& lhs, const Field& rhs) {
    return lhs == dynamic_cast<const FieldData<T, Dt>&>(rhs);
}

using BoolFieldData = FieldData<bool, DataType::BOOL>;
using Int8FieldData = FieldData<int8_t, DataType::INT8>;
using Int16FieldData = FieldData<int16_t, DataType::INT16>;
using Int32FieldData = FieldData<int32_t, DataType::INT32>;
using Int64FieldData = FieldData<int64_t, DataType::INT64>;
using FloatFieldData = FieldData<float, DataType::FLOAT>;
using DoubleFieldData = FieldData<double, DataType::DOUBLE>;
using StringFieldData = FieldData<std::string, DataType::STRING>;
using BinaryVecFieldData = FieldData<std::vector<uint8_t>, DataType::BINARY_VECTOR>;
using FloatVecFieldData = FieldData<std::vector<float>, DataType::FLOAT_VECTOR>;

using BoolFieldDataPtr = std::shared_ptr<BoolFieldData>;
using Int8FieldDataPtr = std::shared_ptr<Int8FieldData>;
using Int16FieldDataPtr = std::shared_ptr<Int16FieldData>;
using Int32FieldDataPtr = std::shared_ptr<Int32FieldData>;
using Int64FieldDataPtr = std::shared_ptr<Int64FieldData>;
using FloatFieldDataPtr = std::shared_ptr<FloatFieldData>;
using DoubleFieldDataPtr = std::shared_ptr<DoubleFieldData>;
using StringFieldDataPtr = std::shared_ptr<StringFieldData>;
using BinaryVecFieldDataPtr = std::shared_ptr<BinaryVecFieldData>;
using FloatVecFieldDataPtr = std::shared_ptr<FloatVecFieldData>;

}  // namespace milvus
