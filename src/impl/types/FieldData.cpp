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

#include "milvus/types/FieldData.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace milvus {

namespace {

template <DataType Dt>
struct DataTypeTraits {
    static const bool is_vector = false;
};

////////////////////////////////////////////////////////////////////////////////////////////////
// the following vector types need to check empty and dimension, sparse vector no need to check
template <>
struct DataTypeTraits<DataType::BINARY_VECTOR> {
    static const bool is_vector = true;
};

template <>
struct DataTypeTraits<DataType::FLOAT_VECTOR> {
    static const bool is_vector = true;
};

template <>
struct DataTypeTraits<DataType::FLOAT16_VECTOR> {
    static const bool is_vector = true;
};

template <>
struct DataTypeTraits<DataType::BFLOAT16_VECTOR> {
    static const bool is_vector = true;
};

template <>
struct DataTypeTraits<DataType::INT8_VECTOR> {
    static const bool is_vector = true;
};
////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, DataType Dt, std::enable_if_t<!DataTypeTraits<Dt>::is_vector, bool> = true>
StatusCode
AddElement(const T& element, std::vector<T>& array) {
    array.push_back(element);
    return StatusCode::OK;
}

template <typename T, DataType Dt, std::enable_if_t<!DataTypeTraits<Dt>::is_vector, bool> = true>
StatusCode
AddElement(T&& element, std::vector<T>& array) {
    array.emplace_back(element);
    return StatusCode::OK;
}

template <typename T, DataType Dt, std::enable_if_t<DataTypeTraits<Dt>::is_vector, bool> = true>
StatusCode
AddElement(const T& element, std::vector<T>& array) {
    if (element.empty()) {
        return StatusCode::VECTOR_IS_EMPTY;
    }

    if (!array.empty() && element.size() != array.at(0).size()) {
        return StatusCode::DIMENSION_NOT_EQUAL;
    }

    array.push_back(element);
    return StatusCode::OK;
}

template <typename T, DataType Dt, std::enable_if_t<DataTypeTraits<Dt>::is_vector, bool> = true>
StatusCode
AddElement(T&& element, std::vector<T>& array) {
    if (element.empty()) {
        return StatusCode::VECTOR_IS_EMPTY;
    }

    if (!array.empty() && element.size() != array.at(0).size()) {
        return StatusCode::DIMENSION_NOT_EQUAL;
    }

    array.emplace_back(element);
    return StatusCode::OK;
}

}  // namespace

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Field class
const std::string&
Field::Name() const {
    return name_;
}

DataType
Field::Type() const {
    return data_type_;
}

DataType
Field::ElementType() const {
    return element_type_;
}

Field::Field(std::string name, DataType data_type) : name_(std::move(name)), data_type_(data_type) {
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// FieldData class
template <typename T, DataType Dt>
FieldData<T, Dt>::FieldData() : Field("", Dt) {
}

template <typename T, DataType Dt>
FieldData<T, Dt>::FieldData(std::string name) : Field(std::move(name), Dt) {
}

template <typename T, DataType Dt>
FieldData<T, Dt>::FieldData(std::string name, const std::vector<T>& data) : Field(std::move(name), Dt) {
    // use "=" instead of constructor because the nlohmann::json constructor does special things
    data_ = data;
}

template <typename T, DataType Dt>
FieldData<T, Dt>::FieldData(std::string name, const std::vector<T>& data, const std::vector<bool>& valid_data)
    : Field(std::move(name), Dt) {
    // use "=" instead of constructor because the nlohmann::json constructor does special things
    data_ = data;
    valid_data_ = valid_data;
}

template <typename T, DataType Dt>
FieldData<T, Dt>::FieldData(std::string name, std::vector<T>&& data) : Field(std::move(name), Dt) {
    // use "=" instead of constructor because the nlohmann::json constructor does special things
    data_ = std::move(data);
}

template <typename T, DataType Dt>
FieldData<T, Dt>::FieldData(std::string name, std::vector<T>&& data, std::vector<bool>&& valid_data)
    : Field(std::move(name), Dt) {
    // use "=" instead of constructor because the nlohmann::json constructor does special things
    data_ = std::move(data);
    valid_data_ = std::move(valid_data);
}

template <typename T, DataType Dt>
StatusCode
FieldData<T, Dt>::Add(const T& element) {
    auto code = AddElement<T, Dt>(element, data_);
    if (code == StatusCode::OK) {
        valid_data_.push_back(true);
    }
    return code;
}

template <typename T, DataType Dt>
StatusCode
FieldData<T, Dt>::Add(T&& element) {
    auto code = AddElement<T, Dt>(element, data_);
    if (code == StatusCode::OK) {
        valid_data_.push_back(true);
    }
    return code;
}

template <typename T, DataType Dt>
StatusCode
FieldData<T, Dt>::AddNull() {
    valid_data_.push_back(false);
    data_.push_back(T());
    return StatusCode::OK;
}

template <typename T, DataType Dt>
StatusCode
FieldData<T, Dt>::Append(const std::vector<T>& elements) {
    data_.reserve(data_.size() + elements.size());
    std::copy(elements.begin(), elements.end(), std::back_inserter(data_));
    for (auto i = 0; i < elements.size(); i++) {
        valid_data_.push_back(true);
    }
    return StatusCode::OK;
}

template <typename T, DataType Dt>
size_t
FieldData<T, Dt>::Count() const {
    return data_.size();
}

template <typename T, DataType Dt>
void
FieldData<T, Dt>::Reserve(size_t count) {
    data_.reserve(count);
    valid_data_.reserve(count);
}

template <typename T, DataType Dt>
const std::vector<T>&
FieldData<T, Dt>::Data() const {
    return data_;
}

template <typename T, DataType Dt>
T
FieldData<T, Dt>::Value(size_t i) const {
    return data_.at(i);
}

template <typename T, DataType Dt>
bool
FieldData<T, Dt>::IsNull(size_t i) const {
    if (i >= valid_data_.size()) {
        return false;
    }
    return !valid_data_.at(i);
}

template <typename T, DataType Dt>
const std::vector<bool>&
FieldData<T, Dt>::ValidData() const {
    return valid_data_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ArrayFieldData class
template <typename T, DataType Et>
ArrayFieldData<T, Et>::ArrayFieldData() : FieldData<ArrayFieldData::ElementT, DataType::ARRAY>() {
    this->element_type_ = Et;
}

template <typename T, DataType Et>
ArrayFieldData<T, Et>::ArrayFieldData(std::string name)
    : FieldData<ArrayFieldData::ElementT, DataType::ARRAY>(std::move(name)) {
    this->element_type_ = Et;
}

template <typename T, DataType Et>
ArrayFieldData<T, Et>::ArrayFieldData(std::string name, const std::vector<ArrayFieldData::ElementT>& data)
    : FieldData<ArrayFieldData::ElementT, DataType::ARRAY>(std::move(name), data) {
    this->element_type_ = Et;
}

template <typename T, DataType Et>
ArrayFieldData<T, Et>::ArrayFieldData(std::string name, const std::vector<ArrayFieldData::ElementT>& data,
                                      const std::vector<bool>& valid_data)
    : FieldData<ArrayFieldData::ElementT, DataType::ARRAY>(std::move(name), data, valid_data) {
    this->element_type_ = Et;
}

template <typename T, DataType Et>
ArrayFieldData<T, Et>::ArrayFieldData(std::string name, std::vector<ArrayFieldData::ElementT>&& data)
    : FieldData<ArrayFieldData::ElementT, DataType::ARRAY>(std::move(name), data) {
    this->element_type_ = Et;
}

template <typename T, DataType Et>
ArrayFieldData<T, Et>::ArrayFieldData(std::string name, std::vector<ArrayFieldData::ElementT>&& data,
                                      std::vector<bool>&& valid_data)
    : FieldData<ArrayFieldData::ElementT, DataType::ARRAY>(std::move(name), data, valid_data) {
    this->element_type_ = Et;
}

template <typename T, DataType Et>
StatusCode
ArrayFieldData<T, Et>::Add(const ArrayFieldData::ElementT& element) {
    this->data_.emplace_back(element);
    return StatusCode::OK;
}

template <typename T, DataType Et>
StatusCode
ArrayFieldData<T, Et>::Add(ArrayFieldData::ElementT&& element) {
    this->data_.emplace_back(element);
    return StatusCode::OK;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// BinaryVecFieldData class
BinaryVecFieldData::BinaryVecFieldData(std::string name)
    : FieldData<std::vector<uint8_t>, DataType::BINARY_VECTOR>(std::move(name)) {
}

BinaryVecFieldData::BinaryVecFieldData(std::string name, const std::vector<std::vector<uint8_t>>& data)
    : FieldData<std::vector<uint8_t>, DataType::BINARY_VECTOR>(std::move(name), data) {
}

BinaryVecFieldData::BinaryVecFieldData(std::string name, const std::vector<std::vector<uint8_t>>& data,
                                       const std::vector<bool>& valid_data)
    : FieldData<std::vector<uint8_t>, DataType::BINARY_VECTOR>(std::move(name), data, valid_data) {
}

BinaryVecFieldData::BinaryVecFieldData(std::string name, std::vector<std::vector<uint8_t>>&& data)
    : FieldData<std::vector<uint8_t>, DataType::BINARY_VECTOR>(std::move(name), std::move(data)) {
}

BinaryVecFieldData::BinaryVecFieldData(std::string name, std::vector<std::vector<uint8_t>>&& data,
                                       std::vector<bool>&& valid_data)
    : FieldData<std::vector<uint8_t>, DataType::BINARY_VECTOR>(std::move(name), std::move(data),
                                                               std::move(valid_data)) {
}

BinaryVecFieldData::BinaryVecFieldData(std::string name, const std::vector<std::string>& data)
    : FieldData<std::vector<uint8_t>, DataType::BINARY_VECTOR>(std::move(name), ToUnsignedChars(data)) {
}

BinaryVecFieldData::BinaryVecFieldData(std::string name, const std::vector<std::string>& data,
                                       const std::vector<bool>& valid_data)
    : FieldData<std::vector<uint8_t>, DataType::BINARY_VECTOR>(std::move(name), ToUnsignedChars(data), valid_data) {
}

BinaryVecFieldData::BinaryVecFieldData(std::string name, std::vector<std::string>&& data)
    : FieldData<std::vector<uint8_t>, DataType::BINARY_VECTOR>(std::move(name), std::move(ToUnsignedChars(data))) {
}

BinaryVecFieldData::BinaryVecFieldData(std::string name, std::vector<std::string>&& data,
                                       std::vector<bool>&& valid_data)
    : FieldData<std::vector<uint8_t>, DataType::BINARY_VECTOR>(std::move(name), std::move(ToUnsignedChars(data)),
                                                               std::move(valid_data)) {
}

std::vector<std::string>
BinaryVecFieldData::DataAsString() const {
    return ToBinaryStrings(data_);
}

StatusCode
BinaryVecFieldData::AddAsString(const std::string& element) {
    auto bin_data = ToUnsignedChars(element);
    return AddElement<std::vector<uint8_t>, DataType::BINARY_VECTOR>(bin_data, data_);
}

StatusCode
BinaryVecFieldData::AddAsString(std::string&& element) {
    auto bin_data = ToUnsignedChars(element);
    return AddElement<std::vector<uint8_t>, DataType::BINARY_VECTOR>(bin_data, data_);
}

std::vector<std::string>
BinaryVecFieldData::ToBinaryStrings(const std::vector<std::vector<uint8_t>>& data) {
    std::vector<std::string> ret;
    ret.reserve(data.size());
    for (const auto& item : data) {
        ret.emplace_back(std::move(ToBinaryString(item)));
    }
    return std::move(ret);
}

std::string
BinaryVecFieldData::ToBinaryString(const std::vector<uint8_t>& data) {
    return std::move(std::string{data.begin(), data.end()});
}

std::vector<std::vector<uint8_t>>
BinaryVecFieldData::ToUnsignedChars(const std::vector<std::string>& data) {
    std::vector<std::vector<uint8_t>> ret;
    ret.reserve(data.size());
    for (const auto& item : data) {
        ret.emplace_back(item.begin(), item.end());
    }
    return std::move(ret);
}

std::vector<uint8_t>
BinaryVecFieldData::ToUnsignedChars(const std::string& data) {
    std::vector<uint8_t> ret;
    ret.reserve(data.size());
    std::copy(data.begin(), data.end(), std::back_inserter(ret));
    return std::move(ret);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// explicit declare FieldData
template class FieldData<bool, DataType::BOOL>;
template class FieldData<int8_t, DataType::INT8>;
template class FieldData<int16_t, DataType::INT16>;
template class FieldData<int32_t, DataType::INT32>;
template class FieldData<int64_t, DataType::INT64>;
template class FieldData<float, DataType::FLOAT>;
template class FieldData<double, DataType::DOUBLE>;
template class FieldData<std::string, DataType::VARCHAR>;
template class FieldData<nlohmann::json, DataType::JSON>;
template class FieldData<std::vector<uint8_t>, DataType::BINARY_VECTOR>;
template class FieldData<std::vector<float>, DataType::FLOAT_VECTOR>;
template class FieldData<std::map<uint32_t, float>, DataType::SPARSE_FLOAT_VECTOR>;
template class FieldData<std::vector<uint16_t>, DataType::FLOAT16_VECTOR>;
template class FieldData<std::vector<uint16_t>, DataType::BFLOAT16_VECTOR>;
template class FieldData<std::vector<int8_t>, DataType::INT8_VECTOR>;

// declare these classes to avoid compile errors on MacOS
template class FieldData<std::vector<bool>, DataType::ARRAY>;
template class FieldData<std::vector<int8_t>, DataType::ARRAY>;
template class FieldData<std::vector<int16_t>, DataType::ARRAY>;
template class FieldData<std::vector<int32_t>, DataType::ARRAY>;
template class FieldData<std::vector<int64_t>, DataType::ARRAY>;
template class FieldData<std::vector<float>, DataType::ARRAY>;
template class FieldData<std::vector<double>, DataType::ARRAY>;
template class FieldData<std::vector<std::string>, DataType::ARRAY>;

template class ArrayFieldData<bool, DataType::BOOL>;
template class ArrayFieldData<int8_t, DataType::INT8>;
template class ArrayFieldData<int16_t, DataType::INT16>;
template class ArrayFieldData<int32_t, DataType::INT32>;
template class ArrayFieldData<int64_t, DataType::INT64>;
template class ArrayFieldData<float, DataType::FLOAT>;
template class ArrayFieldData<double, DataType::DOUBLE>;
template class ArrayFieldData<std::string, DataType::VARCHAR>;

// for struct field
template class ArrayFieldData<nlohmann::json, DataType::STRUCT>;

}  // namespace milvus
