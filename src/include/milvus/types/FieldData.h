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

#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

#include "../Status.h"
#include "DataType.h"

namespace milvus {
class Field {
 public:
    virtual ~Field() = default;

    /**
     * @brief Get field name.
     */
    const std::string&
    Name() const;

    /**
     * @brief Get field data type.
     */
    DataType
    Type() const;

    /**
     * @brief Get elelemnt type for array field.
     */
    DataType
    ElementType() const;

    /**
     * @brief Total number of field elements.
     */
    virtual size_t
    Count() const = 0;

    /**
     * @brief Pre-allocate a space for number of elements.
     */
    virtual void
    Reserve(size_t count) = 0;

 protected:
    Field(std::string name, DataType data_type);

 protected:
    std::string name_;
    DataType data_type_{DataType::UNKNOWN};
    DataType element_type_{DataType::UNKNOWN};  // only for array field
};

using FieldDataPtr = std::shared_ptr<Field>;

/**
 * @brief Template class represents column-based data of a field. Available inheritance classes: \n
 *  BoolFieldData for boolean scalar field \n
 *  Int8FieldData for 8-bits integer scalar field \n
 *  Int16FieldData for 16-bits integer scalar field \n
 *  Int32FieldData for 32-bits integer scalar field \n
 *  Int64FieldData for 64-bits integer scalar field \n
 *  FloatFieldData for float scalar field \n
 *  DoubleFieldData for double scalar field \n
 *  VarCharFieldData for string scalar field \n
 *  JSONFieldData for json scalar field (supportted in 2.4) \n
 *  BinaryVecFieldData for float vector field \n
 *  FloatVecFieldData for binary vector field \n
 *  SparseFloatVecFieldData for sparse vector field \n
 *  Float16VecFieldData for float16 vector field \n
 *  BFloat16VecFieldData for bfloat16 vector field \n
 */
template <typename T, DataType Dt>
class FieldData : public Field {
 public:
    /**
     * @brief Field element type.
     */
    using ElementT = T;

    /**
     * @brief Constructor.
     */
    FieldData();

    /**
     * @brief Constructor.
     */
    explicit FieldData(std::string name);

    /**
     * @brief Constructor.
     */
    FieldData(std::string name, const std::vector<T>& data);

    /**
     * @brief Constructor.
     */
    FieldData(std::string name, const std::vector<T>& data, const std::vector<bool>& valid_data);

    /**
     * @brief Constructor.
     */
    FieldData(std::string name, std::vector<T>&& data);

    /**
     * @brief Constructor.
     */
    FieldData(std::string name, std::vector<T>&& data, std::vector<bool>&& valid_data);

    /**
     * @brief Add element to field data.
     */
    virtual StatusCode
    Add(const T& element);

    /**
     * @brief Add element to field data.
     */
    virtual StatusCode
    Add(T&& element);

    /**
     * @brief Add a null element to field data.
     */
    virtual StatusCode
    AddNull();

    /**
     * @brief Append elements to field data.
     */
    virtual StatusCode
    Append(const std::vector<T>& elements);

    /**
     * @brief Total number of field elements.
     */
    size_t
    Count() const final;

    /**
     * @brief Pre-allocate a space for number of elements.
     */
    void
    Reserve(size_t count) final;

    /**
     * @brief Field elements array.
     */
    virtual const std::vector<T>&
    Data() const;

    /**
     * @brief Get value by position.
     */
    virtual T
    Value(size_t i) const;

    /**
     * @brief Is this position null value.
     */
    virtual bool
    IsNull(size_t i) const;

    /**
     * @brief Bool array to indicate null or non-null elements.
     */
    virtual const std::vector<bool>&
    ValidData() const;

 protected:
    std::vector<T> data_;
    std::vector<bool> valid_data_;
};

/**
 * @brief Template class represents column-based data of an array field(supportted in 2.4). \n
 *  Available inheritance classes: \n
 *  ArrayBoolFieldData for boolean array field \n
 *  ArrayInt8FieldData for 8-bits integer array field \n
 *  ArrayInt16FieldData for 16-bits integer array field \n
 *  ArrayInt32FieldData for 32-bits integer array field \n
 *  ArrayInt64FieldData for 64-bits integer array field \n
 *  ArrayFloatFieldDataPtr for float array field \n
 *  ArrayDoubleFieldData for double array field \n
 *  ArrayVarCharFieldData for string array field \n
 */
template <typename T, DataType Et>
class ArrayFieldData : public FieldData<std::vector<T>, DataType::ARRAY> {
 public:
    /**
     * @brief Field element type.
     */
    using ElementT = std::vector<T>;

    /**
     * @brief Constructor.
     */
    ArrayFieldData();

    /**
     * @brief Constructor.
     */
    explicit ArrayFieldData(std::string name);

    /**
     * @brief Constructor.
     */
    ArrayFieldData(std::string name, const std::vector<ArrayFieldData::ElementT>& data);

    /**
     * @brief Constructor.
     */
    ArrayFieldData(std::string name, const std::vector<ArrayFieldData::ElementT>& data,
                   const std::vector<bool>& valid_data);

    /**
     * @brief Constructor.
     */
    ArrayFieldData(std::string name, std::vector<ArrayFieldData::ElementT>&& data);

    /**
     * @brief Constructor.
     */
    ArrayFieldData(std::string name, std::vector<ArrayFieldData::ElementT>&& data, std::vector<bool>&& valid_data);

    /**
     * @brief Add element to field data.
     */
    StatusCode
    Add(const ArrayFieldData::ElementT& element) override;

    /**
     * @brief Add element to field data.
     */
    StatusCode
    Add(ArrayFieldData::ElementT&& element) override;
};

class BinaryVecFieldData : public FieldData<std::vector<uint8_t>, DataType::BINARY_VECTOR> {
 public:
    /**
     * @brief Field element type.
     */
    using ElementT = std::vector<uint8_t>;

    /**
     * @brief Constructor.
     */
    explicit BinaryVecFieldData(std::string name);

    /**
     * @brief Constructor.
     */
    BinaryVecFieldData(std::string name, const std::vector<std::vector<uint8_t>>& data);

    /**
     * @brief Constructor.
     */
    BinaryVecFieldData(std::string name, const std::vector<std::vector<uint8_t>>& data,
                       const std::vector<bool>& valid_data);

    /**
     * @brief Constructor.
     */
    BinaryVecFieldData(std::string name, std::vector<std::vector<uint8_t>>&& data);

    /**
     * @brief Constructor.
     */
    BinaryVecFieldData(std::string name, std::vector<std::vector<uint8_t>>&& data, std::vector<bool>&& valid_data);

    /**
     * @brief Extra constructor.
     */
    BinaryVecFieldData(std::string name, const std::vector<std::string>& data);

    /**
     * @brief Extra constructor.
     */
    BinaryVecFieldData(std::string name, const std::vector<std::string>& data, const std::vector<bool>& valid_data);

    /**
     * @brief Extra constructor.
     */
    BinaryVecFieldData(std::string name, std::vector<std::string>&& data);

    /**
     * @brief Extra constructor.
     */
    BinaryVecFieldData(std::string name, std::vector<std::string>&& data, std::vector<bool>&& valid_data);

    /**
     * @brief Extra method to get field elements array.
     */
    std::vector<std::string>
    DataAsString() const;

    /**
     * @brief Extra method to add element to field data.
     */
    StatusCode
    AddAsString(const std::string& element);

    /**
     * @brief Extra method to add element to field data.
     */
    StatusCode
    AddAsString(std::string&& element);

    /**
     * @brief Convert binary vectors to strings.
     */
    static std::vector<std::string>
    ToBinaryStrings(const std::vector<std::vector<uint8_t>>& data);

    /**
     * @brief Convert binary vector to string.
     */
    static std::string
    ToBinaryString(const std::vector<uint8_t>& data);

    /**
     * @brief Convert strings to binary vectors.
     */
    static std::vector<std::vector<uint8_t>>
    ToUnsignedChars(const std::vector<std::string>& data);

    /**
     * @brief Convert string to binary vector.
     */
    static std::vector<uint8_t>
    ToUnsignedChars(const std::string& data);
};

using EntityRow = nlohmann::json;
using EntityRows = std::vector<nlohmann::json>;

using BoolFieldData = FieldData<bool, DataType::BOOL>;
using Int8FieldData = FieldData<int8_t, DataType::INT8>;
using Int16FieldData = FieldData<int16_t, DataType::INT16>;
using Int32FieldData = FieldData<int32_t, DataType::INT32>;
using Int64FieldData = FieldData<int64_t, DataType::INT64>;
using FloatFieldData = FieldData<float, DataType::FLOAT>;
using DoubleFieldData = FieldData<double, DataType::DOUBLE>;
using VarCharFieldData = FieldData<std::string, DataType::VARCHAR>;
using JSONFieldData = FieldData<nlohmann::json, DataType::JSON>;
using FloatVecFieldData = FieldData<std::vector<float>, DataType::FLOAT_VECTOR>;
using SparseFloatVecFieldData = FieldData<std::map<uint32_t, float>, DataType::SPARSE_FLOAT_VECTOR>;
using Float16VecFieldData = FieldData<std::vector<uint16_t>, DataType::FLOAT16_VECTOR>;
using BFloat16VecFieldData = FieldData<std::vector<uint16_t>, DataType::BFLOAT16_VECTOR>;

using ArrayBoolFieldData = ArrayFieldData<bool, DataType::BOOL>;
using ArrayInt8FieldData = ArrayFieldData<int8_t, DataType::INT8>;
using ArrayInt16FieldData = ArrayFieldData<int16_t, DataType::INT16>;
using ArrayInt32FieldData = ArrayFieldData<int32_t, DataType::INT32>;
using ArrayInt64FieldData = ArrayFieldData<int64_t, DataType::INT64>;
using ArrayFloatFieldData = ArrayFieldData<float, DataType::FLOAT>;
using ArrayDoubleFieldData = ArrayFieldData<double, DataType::DOUBLE>;
using ArrayVarCharFieldData = ArrayFieldData<std::string, DataType::VARCHAR>;

using BoolFieldDataPtr = std::shared_ptr<BoolFieldData>;
using Int8FieldDataPtr = std::shared_ptr<Int8FieldData>;
using Int16FieldDataPtr = std::shared_ptr<Int16FieldData>;
using Int32FieldDataPtr = std::shared_ptr<Int32FieldData>;
using Int64FieldDataPtr = std::shared_ptr<Int64FieldData>;
using FloatFieldDataPtr = std::shared_ptr<FloatFieldData>;
using DoubleFieldDataPtr = std::shared_ptr<DoubleFieldData>;
using VarCharFieldDataPtr = std::shared_ptr<VarCharFieldData>;
using JSONFieldDataPtr = std::shared_ptr<JSONFieldData>;
using BinaryVecFieldDataPtr = std::shared_ptr<BinaryVecFieldData>;
using FloatVecFieldDataPtr = std::shared_ptr<FloatVecFieldData>;
using SparseFloatVecFieldDataPtr = std::shared_ptr<SparseFloatVecFieldData>;
using Float16VecFieldDataPtr = std::shared_ptr<Float16VecFieldData>;
using BFloat16VecFieldDataPtr = std::shared_ptr<BFloat16VecFieldData>;

using ArrayBoolFieldDataPtr = std::shared_ptr<ArrayBoolFieldData>;
using ArrayInt8FieldDataPtr = std::shared_ptr<ArrayInt8FieldData>;
using ArrayInt16FieldDataPtr = std::shared_ptr<ArrayInt16FieldData>;
using ArrayInt32FieldDataPtr = std::shared_ptr<ArrayInt32FieldData>;
using ArrayInt64FieldDataPtr = std::shared_ptr<ArrayInt64FieldData>;
using ArrayFloatFieldDataPtr = std::shared_ptr<ArrayFloatFieldData>;
using ArrayDoubleFieldDataPtr = std::shared_ptr<ArrayDoubleFieldData>;
using ArrayVarCharFieldDataPtr = std::shared_ptr<ArrayVarCharFieldData>;

extern template class FieldData<bool, DataType::BOOL>;
extern template class FieldData<int8_t, DataType::INT8>;
extern template class FieldData<int16_t, DataType::INT16>;
extern template class FieldData<int32_t, DataType::INT32>;
extern template class FieldData<int64_t, DataType::INT64>;
extern template class FieldData<float, DataType::FLOAT>;
extern template class FieldData<double, DataType::DOUBLE>;
extern template class FieldData<std::string, DataType::VARCHAR>;
extern template class FieldData<nlohmann::json, DataType::JSON>;
extern template class FieldData<std::vector<uint8_t>, DataType::BINARY_VECTOR>;
extern template class FieldData<std::vector<float>, DataType::FLOAT_VECTOR>;
extern template class FieldData<std::map<uint32_t, float>, DataType::SPARSE_FLOAT_VECTOR>;
extern template class FieldData<std::vector<uint16_t>, DataType::FLOAT16_VECTOR>;
extern template class FieldData<std::vector<uint16_t>, DataType::BFLOAT16_VECTOR>;

extern template class ArrayFieldData<bool, DataType::BOOL>;
extern template class ArrayFieldData<int8_t, DataType::INT8>;
extern template class ArrayFieldData<int16_t, DataType::INT16>;
extern template class ArrayFieldData<int32_t, DataType::INT32>;
extern template class ArrayFieldData<int64_t, DataType::INT64>;
extern template class ArrayFieldData<float, DataType::FLOAT>;
extern template class ArrayFieldData<double, DataType::DOUBLE>;
extern template class ArrayFieldData<std::string, DataType::VARCHAR>;

}  // namespace milvus
