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
#include <milvus/thirdparty/nlohmann/json.hpp>
#include <vector>

#include "../Status.h"
#include "DataType.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Base interface of a columnar field in a query, search or DML result.
 */
class MILVUS_SDK_API Field {
 public:
    virtual ~Field() = default;

    /**
     * @brief Get field name.
     * @return the name.
     */
    const std::string&
    Name() const;

    /**
     * @brief Get field data type.
     * @return the type.
     */
    DataType
    Type() const;

    /**
     * @brief Get the element type for an array field.
     * @return the element type.
     */
    DataType
    ElementType() const;

    /**
     * @brief Total number of field elements.
     * @return the count.
     */
    virtual size_t
    Count() const = 0;

    /**
     * @brief Pre-allocate a space for number of elements.
     * @param [in] count the count.
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
 *  Int8FieldData for 8-bit integer scalar field \n
 *  Int16FieldData for 16-bits integer scalar field \n
 *  Int32FieldData for 32-bits integer scalar field \n
 *  Int64FieldData for 64-bits integer scalar field \n
 *  FloatFieldData for float scalar field \n
 *  DoubleFieldData for double scalar field \n
 *  VarCharFieldData for string scalar field \n
 * @return the field.
 *  TextFieldData for text scalar field \n
 *  JSONFieldData for JSON scalar field (supported since 2.4) \n
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
     * @param [in] name the name.
     */
    explicit FieldData(std::string name);

    /**
     * @brief Constructor.
     * @param [in] name the name.
     * @param [in] data the data.
     */
    FieldData(std::string name, const std::vector<T>& data);

    /**
     * @brief Constructor.
     * @param [in] name the name.
     * @param [in] data the data.
     * @param [in] valid_data the valid data.
     */
    FieldData(std::string name, const std::vector<T>& data, const std::vector<bool>& valid_data);

    /**
     * @brief Constructor.
     * @param [in] name the name.
     * @param [in] data the data.
     */
    FieldData(std::string name, std::vector<T>&& data);

    /**
     * @brief Constructor.
     * @param [in] name the name.
     * @param [in] data the data.
     * @param [in] valid_data the valid data.
     */
    FieldData(std::string name, std::vector<T>&& data, std::vector<bool>&& valid_data);

    /**
     * @brief Add element to field data.
     * @param [in] element the element.
     */
    virtual StatusCode
    Add(const T& element);

    /**
     * @brief Add element to field data.
     * @param [in] element the element.
     */
    virtual StatusCode
    Add(T&& element);

    /**
     * @brief Add a null element to field data.
     * @return the add null.
     */
    virtual StatusCode
    AddNull();

    /**
     * @brief Append elements to field data.
     * @param [in] elements the elements.
     */
    virtual StatusCode
    Append(const std::vector<T>& elements);

    /**
     * @brief Append elements and their validity metadata to field data.
     * An empty validity array means every element is valid. Otherwise, its size must match the elements array,
     * where true marks a valid value and false marks a null value.
     * @param [in] elements the elements.
     * @param [in] valid_data the valid data.
     */
    StatusCode
    Append(const std::vector<T>& elements, const std::vector<bool>& valid_data);

    /**
     * @brief Total number of field elements.
     * @return the count.
     */
    size_t
    Count() const final;

    /**
     * @brief Pre-allocate a space for number of elements.
     * @param [in] count the count.
     */
    void
    Reserve(size_t count) final;

    /**
     * @brief Field elements array.
     * @return the data.
     */
    virtual const std::vector<T>&
    Data() const;

    /**
     * @brief Get value by position.
     * @param [in] i the i.
     */
    virtual T
    Value(size_t i) const;

    /**
     * @brief Is this position null value.
     * @param [in] i the i.
     */
    virtual bool
    IsNull(size_t i) const;

    /**
     * @brief Bool array to indicate null or non-null elements.
     * @return the valid data.
     */
    virtual const std::vector<bool>&
    ValidData() const;

 protected:
    std::vector<T> data_;
    std::vector<bool> valid_data_;
};

/**
 * @brief Template class representing column-based data for an array field (supported since 2.4). \n
 *  Available inheritance classes: \n
 *  ArrayBoolFieldData for boolean array field \n
 *  ArrayInt8FieldData for 8-bit integer array field \n
 *  ArrayInt16FieldData for 16-bits integer array field \n
 *  ArrayInt32FieldData for 32-bits integer array field \n
 *  ArrayInt64FieldData for 64-bits integer array field \n
 *  ArrayFloatFieldDataPtr for float array field \n
 *  ArrayDoubleFieldData for double array field \n
 *  ArrayVarCharFieldData for string array field \n
 *  ArrayTextFieldData for text array field \n
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
     * @param [in] name the name.
     */
    explicit ArrayFieldData(std::string name);

    /**
     * @brief Constructor.
     * @param [in] name the name.
     * @param [in] data the data.
     */
    ArrayFieldData(std::string name, const std::vector<ArrayFieldData::ElementT>& data);

    /**
     * @brief Constructor.
     */
    ArrayFieldData(std::string name, const std::vector<ArrayFieldData::ElementT>& data,
                   const std::vector<bool>& valid_data);

    /**
     * @brief Constructor.
     * @param [in] name the name.
     * @param [in] data the data.
     */
    ArrayFieldData(std::string name, std::vector<ArrayFieldData::ElementT>&& data);

    /**
     * @brief Constructor.
     * @param [in] name the name.
     * @param [in] data the data.
     * @param [in] valid_data the valid data.
     */
    ArrayFieldData(std::string name, std::vector<ArrayFieldData::ElementT>&& data, std::vector<bool>&& valid_data);

    /**
     * @brief Add element to field data.
     * @param [in] element the element.
     */
    StatusCode
    Add(const ArrayFieldData::ElementT& element) override;

    /**
     * @brief Add element to field data.
     * @param [in] element the element.
     */
    StatusCode
    Add(ArrayFieldData::ElementT&& element) override;
};

/**
 * @brief Field data of binary vectors (DataType::BINARY_VECTOR).
 */
class MILVUS_SDK_API BinaryVecFieldData : public FieldData<std::vector<uint8_t>, DataType::BINARY_VECTOR> {
 public:
    /**
     * @brief Field element type.
     */
    using ElementT = std::vector<uint8_t>;

    /**
     * @brief Constructor.
     * @param [in] name the name.
     */
    explicit BinaryVecFieldData(std::string name);

    /**
     * @brief Constructor.
     * @param [in] name the name.
     * @param [in] data the data.
     */
    BinaryVecFieldData(std::string name, const std::vector<std::vector<uint8_t>>& data);

    /**
     * @brief Constructor.
     */
    BinaryVecFieldData(std::string name, const std::vector<std::vector<uint8_t>>& data,
                       const std::vector<bool>& valid_data);

    /**
     * @brief Constructor.
     * @param [in] name the name.
     * @param [in] data the data.
     */
    BinaryVecFieldData(std::string name, std::vector<std::vector<uint8_t>>&& data);

    /**
     * @brief Constructor.
     * @param [in] name the name.
     * @param [in] data the data.
     * @param [in] valid_data the valid data.
     */
    BinaryVecFieldData(std::string name, std::vector<std::vector<uint8_t>>&& data, std::vector<bool>&& valid_data);

    /**
     * @brief Extra constructor.
     * @param [in] name the name.
     * @param [in] data the data.
     */
    BinaryVecFieldData(std::string name, const std::vector<std::string>& data);

    /**
     * @brief Extra constructor.
     * @param [in] name the name.
     * @param [in] data the data.
     * @param [in] valid_data the valid data.
     */
    BinaryVecFieldData(std::string name, const std::vector<std::string>& data, const std::vector<bool>& valid_data);

    /**
     * @brief Extra constructor.
     * @param [in] name the name.
     * @param [in] data the data.
     */
    BinaryVecFieldData(std::string name, std::vector<std::string>&& data);

    /**
     * @brief Extra constructor.
     * @param [in] name the name.
     * @param [in] data the data.
     * @param [in] valid_data the valid data.
     */
    BinaryVecFieldData(std::string name, std::vector<std::string>&& data, std::vector<bool>&& valid_data);

    /**
     * @brief Extra method to get field elements array.
     * @return the data as string.
     */
    std::vector<std::string>
    DataAsString() const;

    /**
     * @brief Extra method to add element to field data.
     * @param [in] element the element.
     */
    StatusCode
    AddAsString(const std::string& element);

    /**
     * @brief Extra method to add element to field data.
     * @param [in] element the element.
     */
    StatusCode
    AddAsString(std::string&& element);

    /**
     * @brief Convert binary vectors to strings.
     * @param [in] data the data.
     */
    static std::vector<std::string>
    ToBinaryStrings(const std::vector<std::vector<uint8_t>>& data);

    /**
     * @brief Convert binary vector to string.
     * @param [in] data the data.
     */
    static std::string
    ToBinaryString(const std::vector<uint8_t>& data);

    /**
     * @brief Convert strings to binary vectors.
     * @param [in] data the data.
     */
    static std::vector<std::vector<uint8_t>>
    ToUnsignedChars(const std::vector<std::string>& data);

    /**
     * @brief Convert string to binary vector.
     * @param [in] data the data.
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
using GeometryFieldData = VarCharFieldData;     // geometry field data is passed as string
using TextFieldData = VarCharFieldData;         // text field data is passed as string
using TimestamptzFieldData = VarCharFieldData;  // timestamptz field data is passed as string
using JSONFieldData = FieldData<nlohmann::json, DataType::JSON>;
using FloatVecFieldData = FieldData<std::vector<float>, DataType::FLOAT_VECTOR>;
using SparseFloatVecFieldData = FieldData<std::map<uint32_t, float>, DataType::SPARSE_FLOAT_VECTOR>;
using Float16VecFieldData = FieldData<std::vector<uint16_t>, DataType::FLOAT16_VECTOR>;
using BFloat16VecFieldData = FieldData<std::vector<uint16_t>, DataType::BFLOAT16_VECTOR>;
using Int8VecFieldData = FieldData<std::vector<int8_t>, DataType::INT8_VECTOR>;

using ArrayBoolFieldData = ArrayFieldData<bool, DataType::BOOL>;
using ArrayInt8FieldData = ArrayFieldData<int8_t, DataType::INT8>;
using ArrayInt16FieldData = ArrayFieldData<int16_t, DataType::INT16>;
using ArrayInt32FieldData = ArrayFieldData<int32_t, DataType::INT32>;
using ArrayInt64FieldData = ArrayFieldData<int64_t, DataType::INT64>;
using ArrayFloatFieldData = ArrayFieldData<float, DataType::FLOAT>;
using ArrayDoubleFieldData = ArrayFieldData<double, DataType::DOUBLE>;
using ArrayVarCharFieldData = ArrayFieldData<std::string, DataType::VARCHAR>;
using ArrayTextFieldData = ArrayVarCharFieldData;  // text array element data is passed as string

using StructFieldData = ArrayFieldData<nlohmann::json, DataType::STRUCT>;

using BoolFieldDataPtr = std::shared_ptr<BoolFieldData>;
using Int8FieldDataPtr = std::shared_ptr<Int8FieldData>;
using Int16FieldDataPtr = std::shared_ptr<Int16FieldData>;
using Int32FieldDataPtr = std::shared_ptr<Int32FieldData>;
using Int64FieldDataPtr = std::shared_ptr<Int64FieldData>;
using FloatFieldDataPtr = std::shared_ptr<FloatFieldData>;
using DoubleFieldDataPtr = std::shared_ptr<DoubleFieldData>;
using VarCharFieldDataPtr = std::shared_ptr<VarCharFieldData>;
using GeometryFieldDataPtr = VarCharFieldDataPtr;     // geometry field data is passed as string
using TextFieldDataPtr = VarCharFieldDataPtr;         // text field data is passed as string
using TimestamptzFieldDataPtr = VarCharFieldDataPtr;  // timestamptz field data is passed as string
using JSONFieldDataPtr = std::shared_ptr<JSONFieldData>;
using BinaryVecFieldDataPtr = std::shared_ptr<BinaryVecFieldData>;
using FloatVecFieldDataPtr = std::shared_ptr<FloatVecFieldData>;
using SparseFloatVecFieldDataPtr = std::shared_ptr<SparseFloatVecFieldData>;
using Float16VecFieldDataPtr = std::shared_ptr<Float16VecFieldData>;
using BFloat16VecFieldDataPtr = std::shared_ptr<BFloat16VecFieldData>;
using Int8VecFieldDataPtr = std::shared_ptr<Int8VecFieldData>;

using ArrayBoolFieldDataPtr = std::shared_ptr<ArrayBoolFieldData>;
using ArrayInt8FieldDataPtr = std::shared_ptr<ArrayInt8FieldData>;
using ArrayInt16FieldDataPtr = std::shared_ptr<ArrayInt16FieldData>;
using ArrayInt32FieldDataPtr = std::shared_ptr<ArrayInt32FieldData>;
using ArrayInt64FieldDataPtr = std::shared_ptr<ArrayInt64FieldData>;
using ArrayFloatFieldDataPtr = std::shared_ptr<ArrayFloatFieldData>;
using ArrayDoubleFieldDataPtr = std::shared_ptr<ArrayDoubleFieldData>;
using ArrayVarCharFieldDataPtr = std::shared_ptr<ArrayVarCharFieldData>;
using ArrayTextFieldDataPtr = std::shared_ptr<ArrayTextFieldData>;

using StructFieldDataPtr = std::shared_ptr<StructFieldData>;

extern template class MILVUS_SDK_API FieldData<bool, DataType::BOOL>;
extern template class MILVUS_SDK_API FieldData<int8_t, DataType::INT8>;
extern template class MILVUS_SDK_API FieldData<int16_t, DataType::INT16>;
extern template class MILVUS_SDK_API FieldData<int32_t, DataType::INT32>;
extern template class MILVUS_SDK_API FieldData<int64_t, DataType::INT64>;
extern template class MILVUS_SDK_API FieldData<float, DataType::FLOAT>;
extern template class MILVUS_SDK_API FieldData<double, DataType::DOUBLE>;
extern template class MILVUS_SDK_API FieldData<std::string, DataType::VARCHAR>;
extern template class MILVUS_SDK_API FieldData<nlohmann::json, DataType::JSON>;
extern template class MILVUS_SDK_API FieldData<std::vector<uint8_t>, DataType::BINARY_VECTOR>;
extern template class MILVUS_SDK_API FieldData<std::vector<float>, DataType::FLOAT_VECTOR>;
extern template class MILVUS_SDK_API FieldData<std::map<uint32_t, float>, DataType::SPARSE_FLOAT_VECTOR>;
extern template class MILVUS_SDK_API FieldData<std::vector<uint16_t>, DataType::FLOAT16_VECTOR>;
extern template class MILVUS_SDK_API FieldData<std::vector<uint16_t>, DataType::BFLOAT16_VECTOR>;
extern template class MILVUS_SDK_API FieldData<std::vector<int8_t>, DataType::INT8_VECTOR>;

extern template class MILVUS_SDK_API ArrayFieldData<bool, DataType::BOOL>;
extern template class MILVUS_SDK_API ArrayFieldData<int8_t, DataType::INT8>;
extern template class MILVUS_SDK_API ArrayFieldData<int16_t, DataType::INT16>;
extern template class MILVUS_SDK_API ArrayFieldData<int32_t, DataType::INT32>;
extern template class MILVUS_SDK_API ArrayFieldData<int64_t, DataType::INT64>;
extern template class MILVUS_SDK_API ArrayFieldData<float, DataType::FLOAT>;
extern template class MILVUS_SDK_API ArrayFieldData<double, DataType::DOUBLE>;
extern template class MILVUS_SDK_API ArrayFieldData<std::string, DataType::VARCHAR>;

// for struct field
extern template class MILVUS_SDK_API ArrayFieldData<nlohmann::json, DataType::STRUCT>;

}  // namespace milvus
