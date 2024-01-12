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

namespace milvus {

namespace {

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

template <typename T, DataType Dt, std::enable_if_t<!DataTypeTraits<Dt>::is_vector, bool> = true>
StatusCode
AddElement(const T& element, std::vector<T>& array) {
    array.push_back(element);
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

    array.emplace_back(element);
    return StatusCode::OK;
}

}  // namespace

const std::string&
Field::Name() const {
    return name_;
}

DataType
Field::Type() const {
    return data_type_;
}

Field::Field(std::string name, DataType data_type) : name_(std::move(name)), data_type_(data_type) {
}

template <typename T, DataType Dt>
FieldData<T, Dt>::FieldData() : Field("", Dt) {
}

template <typename T, DataType Dt>
FieldData<T, Dt>::FieldData(std::string name) : Field(std::move(name), Dt) {
}

template <typename T, DataType Dt>
FieldData<T, Dt>::FieldData(std::string name, const std::vector<T>& data) : Field(std::move(name), Dt), data_{data} {
}

template <typename T, DataType Dt>
FieldData<T, Dt>::FieldData(std::string name, std::vector<T>&& data)
    : Field(std::move(name), Dt), data_{std::move(data)} {
}

template <typename T, DataType Dt>
StatusCode
FieldData<T, Dt>::Add(const T& element) {
    return AddElement<T, Dt>(element, data_);
}

template <typename T, DataType Dt>
StatusCode
FieldData<T, Dt>::Add(T&& element) {
    return AddElement<T, Dt>(std::move(element), data_);
}

template <typename T, DataType Dt>
size_t
FieldData<T, Dt>::Count() const {
    return data_.size();
}

template <typename T, DataType Dt>
const std::vector<T>&
FieldData<T, Dt>::Data() const {
    return data_;
}

template <typename T, DataType Dt>
std::vector<T>&
FieldData<T, Dt>::Data() {
    return data_;
}

BinaryVecFieldData::BinaryVecFieldData() : FieldData<std::string, DataType::BINARY_VECTOR>() {
}

BinaryVecFieldData::BinaryVecFieldData(std::string name)
    : FieldData<std::string, DataType::BINARY_VECTOR>(std::move(name)) {
}

BinaryVecFieldData::BinaryVecFieldData(std::string name, const std::vector<std::string>& data)
    : FieldData<std::string, DataType::BINARY_VECTOR>(std::move(name), data) {
}

BinaryVecFieldData::BinaryVecFieldData(std::string name, std::vector<std::string>&& data)
    : FieldData<std::string, DataType::BINARY_VECTOR>(std::move(name), std::move(data)) {
}

BinaryVecFieldData::BinaryVecFieldData(std::string name, const std::vector<std::vector<uint8_t>>& data)
    : FieldData<std::string, DataType::BINARY_VECTOR>(std::move(name), CreateBinaryStrings(data)) {
}

const std::vector<std::string>&
BinaryVecFieldData::Data() const {
    return data_;
}

std::vector<std::string>&
BinaryVecFieldData::Data() {
    return data_;
}

std::vector<std::vector<uint8_t>>
BinaryVecFieldData::DataAsUnsignedChars() const {
    std::vector<std::vector<uint8_t>> ret;
    ret.reserve(data_.size());
    for (const auto& item : data_) {
        ret.emplace_back(item.begin(), item.end());
    }
    return ret;
}

StatusCode
BinaryVecFieldData::Add(const std::string& element) {
    return AddElement<std::string, DataType::BINARY_VECTOR>(element, data_);
}

StatusCode
BinaryVecFieldData::Add(std::string&& element) {
    return AddElement<std::string, DataType::BINARY_VECTOR>(element, data_);
}

StatusCode
BinaryVecFieldData::Add(const std::vector<uint8_t>& element) {
    return Add(std::string{element.begin(), element.end()});
}

std::vector<std::string>
BinaryVecFieldData::CreateBinaryStrings(const std::vector<std::vector<uint8_t>>& data) {
    std::vector<std::string> ret;
    ret.reserve(data.size());
    for (const auto& item : data) {
        ret.emplace_back(item.begin(), item.end());
    }
    return ret;
}

std::string
BinaryVecFieldData::CreateBinaryString(const std::vector<uint8_t>& data) {
    return std::string{data.begin(), data.end()};
}

// explicit declare FieldData
template class FieldData<bool, DataType::BOOL>;
template class FieldData<int8_t, DataType::INT8>;
template class FieldData<int16_t, DataType::INT16>;
template class FieldData<int32_t, DataType::INT32>;
template class FieldData<int64_t, DataType::INT64>;
template class FieldData<float, DataType::FLOAT>;
template class FieldData<double, DataType::DOUBLE>;
template class FieldData<std::string, DataType::VARCHAR>;
template class FieldData<std::string, DataType::BINARY_VECTOR>;
template class FieldData<std::vector<float>, DataType::FLOAT_VECTOR>;

}  // namespace milvus
