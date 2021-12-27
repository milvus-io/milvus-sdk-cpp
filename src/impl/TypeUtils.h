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

#include "milvus.pb.h"
#include "types/FieldData.h"

namespace milvus {

bool
operator==(const proto::schema::FieldData& lhs, const BoolFieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const Int8FieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const Int16FieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const Int32FieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const Int64FieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const FloatFieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const DoubleFieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const StringFieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const BinaryVecFieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const FloatVecFieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const proto::schema::FieldData& rhs);

proto::schema::DataType
DataTypeCast(DataType type);

proto::schema::VectorField*
CreateDataField(const BinaryVecFieldData& field);

proto::schema::VectorField*
CreateDataField(const FloatVecFieldData& field);

proto::schema::ScalarField*
CreateDataField(const BoolFieldData& field);

proto::schema::ScalarField*
CreateDataField(const Int8FieldData& field);

proto::schema::ScalarField*
CreateDataField(const Int16FieldData& field);

proto::schema::ScalarField*
CreateDataField(const Int32FieldData& field);

proto::schema::ScalarField*
CreateDataField(const Int64FieldData& field);

proto::schema::ScalarField*
CreateDataField(const FloatFieldData& field);

proto::schema::ScalarField*
CreateDataField(const DoubleFieldData& field);

proto::schema::ScalarField*
CreateDataField(const StringFieldData& field);

proto::schema::FieldData
CreateDataField(const Field& field);

template <typename T, typename VectorData>
std::vector<std::vector<T>>
BuildFieldDataVectors(int64_t dim, const VectorData& vector_data) {
    std::vector<std::vector<T>> data{};
    const auto row_count = vector_data.size() / dim;
    data.reserve(row_count);
    auto cursor = vector_data.begin();
    while (cursor != vector_data.end()) {
        std::vector<T> item{};
        item.reserve(dim);
        std::copy_n(cursor, dim, std::back_inserter(item));
        data.emplace_back(std::move(item));
        std::advance(cursor, dim);
    }
    return data;
}

template <typename T, typename ScalarData>
std::vector<T>
BuildFieldDataScalars(const ScalarData& scalar_data) {
    std::vector<T> data{};
    data.reserve(scalar_data.size());
    std::copy(scalar_data.begin(), scalar_data.end(), std::back_inserter(data));
    return data;
}

FieldDataPtr
CreateFieldData(const proto::schema::FieldData& field_data);

}  // namespace milvus
