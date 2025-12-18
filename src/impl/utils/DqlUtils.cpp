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

#include "DqlUtils.h"

#include <set>

#include "./Constants.h"
#include "./DmlUtils.h"
#include "./GtsDict.h"
#include "./TypeUtils.h"
#include "milvus/types/Constants.h"
#include "milvus/utils/FP16.h"

namespace milvus {

SparseFloatVecFieldData::ElementT
DecodeSparseFloatVector(std::string& bytes) {
    if (bytes.size() % 8 != 0) {
        throw std::runtime_error("Unexpected binary string is received from server side!");
    }

    // indices are uint32_t type but the protobuf only has int32/int64, so we use int32 to store
    // the binary of uint32_t as both of them are 4 bits width.
    // value type is float 4 bits width, each pair of index/value is 8 bits, the binary length is N * 8 bits.
    size_t count = bytes.size() / 8;
    SparseFloatVecFieldData::ElementT sparse{};
    for (size_t i = 0; i < count; i++) {
        uint32_t index = 0;
        std::memcpy(&index, &bytes[i * 8], sizeof(uint32_t));
        float value = 0.0;
        std::memcpy(&value, &bytes[i * 8 + 4], sizeof(float));
        sparse.insert(std::make_pair(index, value));
    }

    return sparse;
}

std::vector<SparseFloatVecFieldData::ElementT>
BuildFieldDataSparseVectors(const google::protobuf::RepeatedPtrField<std::string>& vector_data, size_t offset,
                            size_t count) {
    std::vector<SparseFloatVecFieldData::ElementT> data{};
    if (vector_data.empty() || offset > vector_data.size() || count == 0) {
        return data;
    }
    if (offset + count > vector_data.size()) {
        count = vector_data.size() - offset;
    }
    data.reserve(count);
    auto cursor = vector_data.begin();
    std::advance(cursor, offset);
    auto end = cursor;
    std::advance(end, count);
    while (cursor != end) {
        std::string bytes = *cursor;
        data.emplace_back(std::move(DecodeSparseFloatVector(bytes)));
        cursor++;
    }
    return data;
}

template <typename T, typename V>
std::vector<std::vector<T>>
BuildFieldDataVectors(size_t bytes_per_vec, const V* vectors_data, size_t data_len, size_t offset, size_t count) {
    std::vector<std::vector<T>> data{};
    if (bytes_per_vec == 0 || data_len == 0 || count == 0) {
        return data;
    }
    size_t data_cnt = data_len * sizeof(V) / bytes_per_vec;
    if (offset > data_cnt) {
        return data;
    }
    if (offset + count > data_cnt) {
        count = data_cnt - offset;
    }
    data.reserve(count);
    size_t t_len = bytes_per_vec / sizeof(T);
    size_t v_len = bytes_per_vec / sizeof(V);
    for (size_t i = offset; i < offset + count; i++) {
        std::vector<T> item{};
        item.resize(t_len);
        std::memcpy(item.data(), vectors_data + i * v_len, bytes_per_vec);
        data.emplace_back(std::move(item));
    }
    return data;
}

template <typename T, typename ScalarData>
std::vector<T>
BuildFieldDataScalars(const ScalarData& scalar_data, size_t offset, size_t count) {
    std::vector<T> data{};
    if (offset >= scalar_data.size()) {
        return data;
    }
    data.reserve(count);
    auto begin = scalar_data.begin();
    std::advance(begin, offset);
    auto end = begin;
    if (offset + count > scalar_data.size()) {
        count = scalar_data.size() - offset;
    }
    std::advance(end, count);
    std::copy(begin, end, std::back_inserter(data));
    return data;
}

template <typename T, typename ScalarData>
std::vector<T>
BuildFieldDataScalars(const ScalarData& scalar_data) {
    return BuildFieldDataScalars<T>(scalar_data, 0, scalar_data.size());
}

Status
BuildMilvusArrayFieldData(const std::string& name, const proto::schema::ArrayArray& array_field,
                          std::vector<bool>&& valid_data, size_t offset, size_t count, FieldDataPtr& field_data) {
    field_data = nullptr;
    const auto& scalars_data = array_field.data();
    const auto head = scalars_data.begin();
    const auto total = scalars_data.size();
    auto begin = scalars_data.begin();
    std::advance(begin, offset);
    auto end = begin;
    std::advance(end, count);

    std::string field_name = name;
    proto::schema::DataType element_type = array_field.element_type();
    switch (element_type) {
        case proto::schema::DataType::Bool: {
            std::vector<ArrayBoolFieldData::ElementT> arr;
            for (; begin != end && std::distance(head, begin) < total; begin++) {
                arr.emplace_back(std::move(BuildFieldDataScalars<bool>((*begin).bool_data().data())));
            }
            field_data =
                std::make_shared<ArrayBoolFieldData>(std::move(field_name), std::move(arr), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::Int8: {
            std::vector<ArrayInt8FieldData::ElementT> arr;
            for (; begin != end && std::distance(head, begin) < total; begin++) {
                arr.emplace_back(std::move(BuildFieldDataScalars<int8_t>((*begin).int_data().data())));
            }
            field_data =
                std::make_shared<ArrayInt8FieldData>(std::move(field_name), std::move(arr), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::Int16: {
            std::vector<ArrayInt16FieldData::ElementT> arr;
            for (; begin != end && std::distance(head, begin) < total; begin++) {
                arr.emplace_back(std::move(BuildFieldDataScalars<int16_t>((*begin).int_data().data())));
            }
            field_data =
                std::make_shared<ArrayInt16FieldData>(std::move(field_name), std::move(arr), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::Int32: {
            std::vector<ArrayInt32FieldData::ElementT> arr;
            for (; begin != end && std::distance(head, begin) < total; begin++) {
                arr.emplace_back(std::move(BuildFieldDataScalars<int32_t>((*begin).int_data().data())));
            }
            field_data =
                std::make_shared<ArrayInt32FieldData>(std::move(field_name), std::move(arr), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::Int64: {
            std::vector<ArrayInt64FieldData::ElementT> arr;
            for (; begin != end && std::distance(head, begin) < total; begin++) {
                arr.emplace_back(std::move(BuildFieldDataScalars<int64_t>((*begin).long_data().data())));
            }
            field_data =
                std::make_shared<ArrayInt64FieldData>(std::move(field_name), std::move(arr), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::Float: {
            std::vector<ArrayFloatFieldData::ElementT> arr;
            for (; begin != end && std::distance(head, begin) < total; begin++) {
                arr.emplace_back(std::move(BuildFieldDataScalars<float>((*begin).float_data().data())));
            }
            field_data =
                std::make_shared<ArrayFloatFieldData>(std::move(field_name), std::move(arr), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::Double: {
            std::vector<ArrayDoubleFieldData::ElementT> arr;
            for (; begin != end && std::distance(head, begin) < total; begin++) {
                arr.emplace_back(std::move(BuildFieldDataScalars<double>((*begin).double_data().data())));
            }
            field_data =
                std::make_shared<ArrayDoubleFieldData>(std::move(field_name), std::move(arr), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::VarChar:
        case proto::schema::DataType::Timestamptz: {
            std::vector<ArrayVarCharFieldData::ElementT> arr;
            for (; begin != end && std::distance(head, begin) < total; begin++) {
                arr.emplace_back(std::move(BuildFieldDataScalars<std::string>((*begin).string_data().data())));
            }
            field_data =
                std::make_shared<ArrayVarCharFieldData>(std::move(field_name), std::move(arr), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::Geometry: {
            std::vector<ArrayVarCharFieldData::ElementT> arr;
            for (; begin != end && std::distance(head, begin) < total; begin++) {
                arr.emplace_back(std::move(BuildFieldDataScalars<std::string>((*begin).geometry_wkt_data().data())));
            }
            field_data =
                std::make_shared<ArrayVarCharFieldData>(std::move(field_name), std::move(arr), std::move(valid_data));
            return Status::OK();
        }
        default:
            return {StatusCode::NOT_SUPPORTED, "Unsupported array element type: " + std::to_string(element_type)};
    }
}

Status
GetValidData(const google::protobuf::RepeatedField<bool>& proto_valid, size_t offset, size_t count,
             std::vector<bool>& valid_data) {
    valid_data.clear();
    if (proto_valid.empty()) {
        return Status::OK();
    }
    const bool* valid_begin = proto_valid.data();
    if (valid_begin != nullptr && offset < proto_valid.size()) {
        if (offset + count > proto_valid.size()) {
            count = proto_valid.size() - offset;
        }
        valid_begin = valid_begin + offset;
        valid_data = std::vector<bool>(valid_begin, valid_begin + count);
    }
    return Status::OK();
}

Status
CreateMilvusFieldData(const proto::schema::FieldData& proto_data, size_t offset, size_t count,
                      FieldDataPtr& field_data) {
    field_data = nullptr;
    auto field_type = proto_data.type();
    std::string name = proto_data.field_name();
    const auto& proto_vectors = proto_data.vectors();
    const auto& proto_scalars = proto_data.scalars();
    const auto& proto_valid = proto_data.valid_data();
    std::vector<bool> valid_data;
    GetValidData(proto_valid, offset, count, valid_data);

    switch (field_type) {
        case proto::schema::DataType::BinaryVector: {
            const auto& str = proto_vectors.binary_vector();
            std::vector<BinaryVecFieldData::ElementT> vectors =
                BuildFieldDataVectors<uint8_t, char>(proto_vectors.dim() / 8, str.data(), str.size(), offset, count);
            field_data =
                std::make_shared<BinaryVecFieldData>(std::move(name), std::move(vectors), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::FloatVector: {
            const auto& floats = proto_vectors.float_vector().data();
            std::vector<FloatVecFieldData::ElementT> vectors = BuildFieldDataVectors<float, float>(
                proto_vectors.dim() * 4, floats.data(), floats.size(), offset, count);
            field_data =
                std::make_shared<FloatVecFieldData>(std::move(name), std::move(vectors), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::Float16Vector: {
            const auto& str = proto_vectors.float16_vector();
            std::vector<Float16VecFieldData::ElementT> vectors =
                BuildFieldDataVectors<uint16_t, char>(proto_vectors.dim() * 2, str.data(), str.size(), offset, count);
            field_data =
                std::make_shared<Float16VecFieldData>(std::move(name), std::move(vectors), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::BFloat16Vector: {
            const auto& str = proto_vectors.bfloat16_vector();
            std::vector<BFloat16VecFieldData::ElementT> vectors =
                BuildFieldDataVectors<uint16_t, char>(proto_vectors.dim() * 2, str.data(), str.size(), offset, count);
            field_data =
                std::make_shared<BFloat16VecFieldData>(std::move(name), std::move(vectors), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::SparseFloatVector: {
            std::vector<SparseFloatVecFieldData::ElementT> vectors =
                BuildFieldDataSparseVectors(proto_vectors.sparse_float_vector().contents(), offset, count);
            field_data =
                std::make_shared<SparseFloatVecFieldData>(std::move(name), std::move(vectors), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::Int8Vector: {
            const auto& str = proto_vectors.int8_vector();
            std::vector<Int8VecFieldData::ElementT> vectors =
                BuildFieldDataVectors<int8_t, char>(proto_vectors.dim(), str.data(), str.size(), offset, count);
            field_data = std::make_shared<Int8VecFieldData>(std::move(name), std::move(vectors), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::Bool: {
            std::vector<BoolFieldData::ElementT> values =
                BuildFieldDataScalars<BoolFieldData::ElementT>(proto_scalars.bool_data().data(), offset, count);
            field_data = std::make_shared<BoolFieldData>(std::move(name), std::move(values), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::Int8: {
            std::vector<Int8FieldData::ElementT> values =
                BuildFieldDataScalars<Int8FieldData::ElementT>(proto_scalars.int_data().data(), offset, count);
            field_data = std::make_shared<Int8FieldData>(std::move(name), std::move(values), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::Int16: {
            std::vector<Int16FieldData::ElementT> values =
                BuildFieldDataScalars<Int16FieldData::ElementT>(proto_scalars.int_data().data(), offset, count);
            field_data = std::make_shared<Int16FieldData>(std::move(name), std::move(values), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::Int32: {
            std::vector<Int32FieldData::ElementT> values =
                BuildFieldDataScalars<Int32FieldData::ElementT>(proto_scalars.int_data().data(), offset, count);
            field_data = std::make_shared<Int32FieldData>(std::move(name), std::move(values), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::Int64: {
            std::vector<Int64FieldData::ElementT> values =
                BuildFieldDataScalars<Int64FieldData::ElementT>(proto_scalars.long_data().data(), offset, count);
            field_data = std::make_shared<Int64FieldData>(std::move(name), std::move(values), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::Float: {
            std::vector<FloatFieldData::ElementT> values =
                BuildFieldDataScalars<FloatFieldData::ElementT>(proto_scalars.float_data().data(), offset, count);
            field_data = std::make_shared<FloatFieldData>(std::move(name), std::move(values), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::Double: {
            std::vector<DoubleFieldData::ElementT> values =
                BuildFieldDataScalars<DoubleFieldData::ElementT>(proto_scalars.double_data().data(), offset, count);
            field_data = std::make_shared<DoubleFieldData>(std::move(name), std::move(values), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::VarChar:
        case proto::schema::DataType::Timestamptz: {
            std::vector<VarCharFieldData::ElementT> values =
                BuildFieldDataScalars<VarCharFieldData::ElementT>(proto_scalars.string_data().data(), offset, count);
            field_data = std::make_shared<VarCharFieldData>(std::move(name), std::move(values), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::Geometry: {
            std::vector<VarCharFieldData::ElementT> values = BuildFieldDataScalars<VarCharFieldData::ElementT>(
                proto_scalars.geometry_wkt_data().data(), offset, count);
            field_data = std::make_shared<VarCharFieldData>(std::move(name), std::move(values), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::JSON: {
            std::vector<nlohmann::json> objects;
            const auto& scalars_data = proto_scalars.json_data().data();
            for (const auto& s : scalars_data) {
                objects.emplace_back(std::move(nlohmann::json::parse(s)));
            }
            std::vector<JSONFieldData::ElementT> values =
                BuildFieldDataScalars<JSONFieldData::ElementT>(objects, offset, count);
            field_data = std::make_shared<JSONFieldData>(std::move(name), std::move(values), std::move(valid_data));
            return Status::OK();
        }
        case proto::schema::DataType::Array: {
            return BuildMilvusArrayFieldData(name, proto_scalars.array_data(), std::move(valid_data), offset, count,
                                             field_data);
        }
        case proto::schema::DataType::ArrayOfStruct: {
            return ConvertStructFieldData(proto_data, offset, count, field_data);
        }
        default:
            return {StatusCode::NOT_SUPPORTED, "Unsupported field type: " + std::to_string(field_type)};
    }
}

Status
GetFieldDataRowCount(const proto::schema::FieldData& proto_data, size_t& row_count) {
    row_count = 0;
    auto field_type = proto_data.type();
    const auto& proto_vectors = proto_data.vectors();
    const auto& proto_scalars = proto_data.scalars();
    switch (field_type) {
        case proto::schema::DataType::BinaryVector: {
            size_t len = proto_vectors.dim() / 8;
            row_count = proto_vectors.binary_vector().size() / len;
            break;
        }
        case proto::schema::DataType::FloatVector: {
            size_t len = proto_vectors.dim();
            row_count = proto_vectors.float_vector().data().size() / len;
            break;
        }
        case proto::schema::DataType::Float16Vector: {
            size_t in_len = proto_vectors.dim() * 2;
            row_count = proto_vectors.float16_vector().size() / in_len;
            break;
        }
        case proto::schema::DataType::BFloat16Vector: {
            size_t in_len = proto_vectors.dim() * 2;
            row_count = proto_vectors.bfloat16_vector().size() / in_len;
            break;
        }
        case proto::schema::DataType::SparseFloatVector: {
            const auto& content = proto_vectors.sparse_float_vector().contents();
            row_count = content.size();
            break;
        }
        case proto::schema::DataType::Int8Vector: {
            size_t len = proto_vectors.dim();
            row_count = proto_vectors.int8_vector().size() / len;
            break;
        }
        case proto::schema::DataType::Bool: {
            row_count = proto_scalars.bool_data().data().size();
            break;
        }
        case proto::schema::DataType::Int8:
        case proto::schema::DataType::Int16:
        case proto::schema::DataType::Int32: {
            row_count = proto_scalars.int_data().data().size();
            break;
        }
        case proto::schema::DataType::Int64: {
            row_count = proto_scalars.long_data().data().size();
            break;
        }
        case proto::schema::DataType::Float: {
            row_count = proto_scalars.float_data().data().size();
            break;
        }
        case proto::schema::DataType::Double: {
            row_count = proto_scalars.double_data().data().size();
            break;
        }
        case proto::schema::DataType::VarChar:
        case proto::schema::DataType::Timestamptz: {
            row_count = proto_scalars.string_data().data().size();
            break;
        }
        case proto::schema::DataType::Geometry: {
            row_count = proto_scalars.geometry_wkt_data().data().size();
            break;
        }
        case proto::schema::DataType::JSON: {
            row_count = proto_scalars.json_data().data().size();
            break;
        }
        case proto::schema::DataType::Array: {
            row_count = proto_scalars.array_data().data().size();
            break;
        }
        case proto::schema::DataType::ArrayOfVector: {
            row_count = proto_vectors.vector_array().data_size();
            break;
        }
        case proto::schema::DataType::ArrayOfStruct: {
            const auto& struct_arrays = proto_data.struct_arrays();
            if (struct_arrays.fields_size() == 0) {
                return {StatusCode::UNKNOWN_ERROR, "The returned search result contains an empty StructArrayField"};
            }
            return GetFieldDataRowCount(struct_arrays.fields(0), row_count);
            break;
        }
        default:
            return {StatusCode::NOT_SUPPORTED, "Unsupported field type: " + std::to_string(field_type)};
    }
    return Status::OK();
}

Status
CreateMilvusFieldData(const proto::schema::FieldData& proto_data, FieldDataPtr& field_data) {
    auto field_type = proto_data.type();
    size_t row_count = 0;
    auto status = GetFieldDataRowCount(proto_data, row_count);
    if (!status.IsOk()) {
        return status;
    }

    if (field_type == proto::schema::DataType::ArrayOfStruct) {
        return ConvertStructFieldData(proto_data, 0, row_count, field_data);
    } else {
        return CreateMilvusFieldData(proto_data, 0, row_count, field_data);
    }
}

template <typename T>
void
FillStructValue(const FieldDataPtr& array_data, std::vector<std::vector<nlohmann::json>>& structs) {
    std::shared_ptr<T> actual_ptr = std::static_pointer_cast<T>(array_data);
    auto actual_count = array_data->Count();
    for (auto k = 0; k < actual_count; k++) {
        const auto& arr = actual_ptr->Value(k);
        if (structs.size() <= k) {
            structs.emplace_back(std::move(std::vector<nlohmann::json>()));
            structs[k].resize(arr.size());
        }
        for (auto j = 0; j < arr.size(); j++) {
            structs[k][j][array_data->Name()] = arr[j];
        }
    }
}

Status
ConvertStructFieldData(const proto::schema::FieldData& proto_data, size_t offset, size_t count,
                       FieldDataPtr& field_data) {
    auto sturct_name = proto_data.field_name();
    std::vector<std::vector<nlohmann::json>> structs;
    structs.reserve(count);
    const auto& struct_array = proto_data.struct_arrays();
    for (auto i = 0; i < struct_array.fields_size(); i++) {
        const auto& field_data = struct_array.fields(i);
        auto sub_field_name = field_data.field_name();
        auto field_type = field_data.type();
        switch (field_type) {
            case proto::schema::DataType::Array: {
                const auto& proto_scalars = field_data.scalars();
                const auto& proto_valid = field_data.valid_data();
                std::vector<bool> valid_data;
                GetValidData(proto_valid, offset, count, valid_data);

                FieldDataPtr array_data;
                auto status = BuildMilvusArrayFieldData(sub_field_name, proto_scalars.array_data(),
                                                        std::move(valid_data), offset, count, array_data);
                if (!status.IsOk()) {
                    return status;
                }

                switch (array_data->ElementType()) {
                    case DataType::BOOL: {
                        FillStructValue<ArrayBoolFieldData>(array_data, structs);
                        break;
                    }
                    case DataType::INT8: {
                        FillStructValue<ArrayInt8FieldData>(array_data, structs);
                        break;
                    }
                    case DataType::INT16: {
                        FillStructValue<ArrayInt16FieldData>(array_data, structs);
                        break;
                    }
                    case DataType::INT32: {
                        FillStructValue<ArrayInt32FieldData>(array_data, structs);
                        break;
                    }
                    case DataType::INT64: {
                        FillStructValue<ArrayInt64FieldData>(array_data, structs);
                        break;
                    }
                    case DataType::FLOAT: {
                        FillStructValue<ArrayFloatFieldData>(array_data, structs);
                        break;
                    }
                    case DataType::DOUBLE: {
                        FillStructValue<ArrayDoubleFieldData>(array_data, structs);
                        break;
                    }
                    case DataType::VARCHAR: {
                        FillStructValue<ArrayVarCharFieldData>(array_data, structs);
                        break;
                    }
                    default:
                        return {StatusCode::NOT_SUPPORTED,
                                "Unsupported sub field type: " + std::to_string(array_data->ElementType()) +
                                    " for struct field: " + sturct_name};
                }
                break;
            }
            case proto::schema::DataType::ArrayOfVector: {
                const auto& vector_array = field_data.vectors().vector_array();
                if (vector_array.element_type() != proto::schema::DataType::FloatVector) {
                    return {StatusCode::NOT_SUPPORTED,
                            "Unsupported vector field type: " + std::to_string(vector_array.element_type()) +
                                " for struct field: " + sturct_name};
                }

                if (offset >= vector_array.data_size() || count == 0) {
                    break;
                }
                if (offset + count > vector_array.data_size()) {
                    count = vector_array.data_size() - offset;
                }
                for (size_t k = offset; k < offset + count; k++) {
                    const auto& vector_field = vector_array.data(k);
                    const auto& floats = vector_field.float_vector().data();
                    std::vector<std::vector<float>> vectors = BuildFieldDataVectors<float, float>(
                        vector_field.dim() * 4, floats.data(), floats.size(), 0, floats.size());
                    auto num = k - offset;
                    if (structs.size() <= num) {
                        structs.emplace_back(std::move(std::vector<nlohmann::json>()));
                        structs[num].resize(vectors.size());
                    }
                    for (auto j = 0; j < vectors.size(); j++) {
                        structs[num][j][sub_field_name] = vectors[j];
                    }
                }
                break;
            }
            default:
                return {StatusCode::NOT_SUPPORTED, "Unsupported field type: " + std::to_string(field_type)};
        }
    }

    field_data = std::make_shared<StructFieldData>(proto_data.field_name(), std::move(structs));
    return Status::OK();
}

FieldDataPtr
CreateIDField(const std::string& name, const proto::schema::IDs& ids, size_t offset, size_t size) {
    if (ids.has_int_id()) {
        std::vector<int64_t> int_array;
        auto& int_ids = ids.int_id();
        int_array.reserve(size);
        auto it = int_ids.data().begin();
        std::advance(it, offset);
        auto it_end = it;
        std::advance(it_end, size);
        std::copy(it, it_end, std::back_inserter(int_array));
        return std::make_shared<Int64FieldData>(name, std::move(int_array));
    } else {
        std::vector<std::string> str_array;
        auto& str_ids = ids.str_id();
        str_array.reserve(size);
        auto it = str_ids.data().begin();
        std::advance(it, offset);
        auto it_end = it;
        std::advance(it_end, size);
        std::copy(it, it_end, std::back_inserter(str_array));
        return std::make_shared<VarCharFieldData>(name, std::move(str_array));
    }
}

FieldDataPtr
CreateScoreField(const std::string& name, const proto::schema::SearchResultData& data, size_t offset, size_t size) {
    const auto& scores = data.scores();
    std::vector<float> score_values;
    score_values.reserve(size);
    auto it = scores.begin();
    std::advance(it, offset);
    auto it_end = it;
    std::advance(it_end, size);
    std::copy(it, it_end, std::back_inserter(score_values));
    return std::make_shared<FloatFieldData>(name, std::move(score_values));
}

Status
SetTargetVectors(const FieldDataPtr& target, proto::milvus::SearchRequest* rpc_request) {
    // placeholders
    proto::common::PlaceholderGroup placeholder_group;
    auto& placeholder_value = *placeholder_group.add_placeholders();
    placeholder_value.set_tag("$0");
    if (target->Type() == DataType::BINARY_VECTOR) {
        // binary vector
        placeholder_value.set_type(proto::common::PlaceholderType::BinaryVector);
        auto& vectors = dynamic_cast<BinaryVecFieldData&>(*target);
        for (const auto& vector : vectors.Data()) {
            std::string placeholder_data(reinterpret_cast<const char*>(vector.data()), vector.size());
            placeholder_value.add_values(std::move(placeholder_data));
        }
        rpc_request->set_nq(static_cast<int64_t>(vectors.Count()));
    } else if (target->Type() == DataType::FLOAT_VECTOR) {
        // float vector
        placeholder_value.set_type(proto::common::PlaceholderType::FloatVector);
        auto& vectors = dynamic_cast<FloatVecFieldData&>(*target);
        for (const auto& vector : vectors.Data()) {
            std::string placeholder_data(reinterpret_cast<const char*>(vector.data()), vector.size() * sizeof(float));
            placeholder_value.add_values(std::move(placeholder_data));
        }
        rpc_request->set_nq(static_cast<int64_t>(vectors.Count()));
    } else if (target->Type() == DataType::SPARSE_FLOAT_VECTOR) {
        // sparse vector
        placeholder_value.set_type(proto::common::PlaceholderType::SparseFloatVector);
        auto& vectors = dynamic_cast<SparseFloatVecFieldData&>(*target);
        for (const auto& sparse : vectors.Data()) {
            std::string placeholder_data = EncodeSparseFloatVector(sparse);
            placeholder_value.add_values(std::move(placeholder_data));
        }
        rpc_request->set_nq(static_cast<int64_t>(vectors.Count()));
    } else if (target->Type() == DataType::FLOAT16_VECTOR) {
        // float16 vector
        placeholder_value.set_type(proto::common::PlaceholderType::Float16Vector);
        auto& vectors = dynamic_cast<Float16VecFieldData&>(*target);
        for (const auto& vector : vectors.Data()) {
            std::string placeholder_data(reinterpret_cast<const char*>(vector.data()),
                                         vector.size() * sizeof(uint16_t));
            placeholder_value.add_values(std::move(placeholder_data));
        }
        rpc_request->set_nq(static_cast<int64_t>(vectors.Count()));
    } else if (target->Type() == DataType::BFLOAT16_VECTOR) {
        // bfloat16 vector
        placeholder_value.set_type(proto::common::PlaceholderType::BFloat16Vector);
        auto& vectors = dynamic_cast<BFloat16VecFieldData&>(*target);
        for (const auto& vector : vectors.Data()) {
            std::string placeholder_data(reinterpret_cast<const char*>(vector.data()),
                                         vector.size() * sizeof(uint16_t));
            placeholder_value.add_values(std::move(placeholder_data));
        }
        rpc_request->set_nq(static_cast<int64_t>(vectors.Count()));
    } else if (target->Type() == DataType::INT8_VECTOR) {
        // int8 vector
        placeholder_value.set_type(proto::common::PlaceholderType::Int8Vector);
        auto& vectors = dynamic_cast<Int8VecFieldData&>(*target);
        for (const auto& vector : vectors.Data()) {
            std::string placeholder_data(reinterpret_cast<const char*>(vector.data()), vector.size() * sizeof(int8_t));
            placeholder_value.add_values(std::move(placeholder_data));
        }
        rpc_request->set_nq(static_cast<int64_t>(vectors.Count()));
    } else if (target->Type() == DataType::VARCHAR) {
        // BM25
        placeholder_value.set_type(proto::common::PlaceholderType::VarChar);
        auto& texts = dynamic_cast<VarCharFieldData&>(*target);
        for (std::string text : texts.Data()) {
            placeholder_value.add_values(std::move(text));
        }
        rpc_request->set_nq(static_cast<int64_t>(texts.Count()));
    } else {
        return {StatusCode::NOT_SUPPORTED, "Unsupported target type: " + std::to_string(target->Type())};
    }

    rpc_request->set_placeholder_group(std::move(placeholder_group.SerializeAsString()));
    return Status::OK();
}

Status
SetEmbeddingLists(const std::vector<EmbeddingList>& emb_lists, proto::milvus::SearchRequest* rpc_request) {
    // placeholders
    proto::common::PlaceholderGroup placeholder_group;
    auto& placeholder_value = *placeholder_group.add_placeholders();
    placeholder_value.set_tag("$0");

    for (const auto& emb_list : emb_lists) {
        auto target = emb_list.TargetVectors();
        if (target == nullptr) {
            return {StatusCode::INVALID_AGUMENT, "Embedding list is empty"};
        }
        if (target->Type() == DataType::FLOAT_VECTOR) {
            // so far only support float vector
            placeholder_value.set_type(proto::common::PlaceholderType::EmbListFloatVector);
            auto& vectors = dynamic_cast<FloatVecFieldData&>(*target);
            std::string content;
            content.reserve(emb_list.Count() * emb_list.Dim() * 4);
            for (const auto& vector : vectors.Data()) {
                std::string single_content(reinterpret_cast<const char*>(vector.data()), vector.size() * sizeof(float));
                content += single_content;
            }
            rpc_request->set_nq(static_cast<int64_t>(emb_list.Count()));
            placeholder_value.add_values(std::move(content));
        }
    }

    rpc_request->set_nq(static_cast<int64_t>(emb_lists.size()));
    rpc_request->set_placeholder_group(std::move(placeholder_group.SerializeAsString()));
    return Status::OK();
}

void
SetExtraParams(const std::unordered_map<std::string, std::string>& params,
               ::google::protobuf::RepeatedPtrField<proto::common::KeyValuePair>* kv_pairs) {
    // offet/radius/range_filter/nprobe etc.
    // in old milvus versions, all extra params are under "params"
    // in new milvus versions, all extra params are in the top level
    nlohmann::json json_params;
    for (auto& pair : params) {
        proto::common::KeyValuePair kv_pair;
        kv_pair.set_key(pair.first);
        kv_pair.set_value(pair.second);
        kv_pairs->Add(std::move(kv_pair));

        // for radius/range, the value should be a numeric instead a string in the JSON string
        // for example:
        //   '{"radius": "2.5", "range_filter": "0.5"}' is illegal in the server-side
        //   '{"radius": 2.5, "range_filter": 0.5}' is ok
        if (pair.first == RADIUS || pair.first == RANGE_FILTER) {
            json_params[pair.first] = std::stod(pair.second);
        } else {
            json_params[pair.first] = pair.second;
        }
    }
    {
        proto::common::KeyValuePair kv_pair;
        kv_pair.set_key(PARAMS);
        kv_pair.set_value(json_params.dump());
        kv_pairs->Add(std::move(kv_pair));
    }
}

template <typename T>
std::function<nlohmann::json(size_t)>
GenGetter(const FieldDataPtr& field) {
    return [&field](size_t i) {
        // special process float16/bfloat16 vector to float arrays
        if (field->Type() == DataType::FLOAT16_VECTOR || field->Type() == DataType::BFLOAT16_VECTOR) {
            bool is_fp16 = (field->Type() == DataType::FLOAT16_VECTOR);
            std::vector<uint16_t> f16_vec = is_fp16 ? std::static_pointer_cast<Float16VecFieldData>(field)->Value(i)
                                                    : std::static_pointer_cast<BFloat16VecFieldData>(field)->Value(i);
            std::vector<float> f32_vec;
            f32_vec.reserve(f16_vec.size());
            std::transform(f16_vec.begin(), f16_vec.end(), std::back_inserter(f32_vec),
                           [&is_fp16](uint16_t val) { return is_fp16 ? F16toF32(val) : BF16toF32(val); });
            return nlohmann::json(f32_vec);
        } else {
            std::shared_ptr<T> real_field = std::static_pointer_cast<T>(field);
            if (real_field->IsNull(i)) {
                return nlohmann::json();
            } else {
                return nlohmann::json(real_field->Value(i));
            }
        }
    };
}

std::map<std::string, std::function<nlohmann::json(size_t)>>
GenGetters(const std::vector<FieldDataPtr>& fields) {
    std::map<std::string, std::function<nlohmann::json(size_t)>> getters;
    for (const auto& field : fields) {
        DataType dt = field->Type();
        const std::string& name = field->Name();
        switch (dt) {
            case DataType::BOOL: {
                getters.insert(std::make_pair(name, std::move(GenGetter<BoolFieldData>(field))));
                break;
            }
            case DataType::INT8: {
                getters.insert(std::make_pair(name, std::move(GenGetter<Int8FieldData>(field))));
                break;
            }
            case DataType::INT16: {
                getters.insert(std::make_pair(name, std::move(GenGetter<Int16FieldData>(field))));
                break;
            }
            case DataType::INT32: {
                getters.insert(std::make_pair(name, std::move(GenGetter<Int32FieldData>(field))));
                break;
            }
            case DataType::INT64: {
                getters.insert(std::make_pair(name, std::move(GenGetter<Int64FieldData>(field))));
                break;
            }
            case DataType::FLOAT: {
                getters.insert(std::make_pair(name, std::move(GenGetter<FloatFieldData>(field))));
                break;
            }
            case DataType::DOUBLE: {
                getters.insert(std::make_pair(name, std::move(GenGetter<DoubleFieldData>(field))));
                break;
            }
            case DataType::VARCHAR:
            case DataType::GEOMETRY:
            case DataType::TIMESTAMPTZ: {
                getters.insert(std::make_pair(name, std::move(GenGetter<VarCharFieldData>(field))));
                break;
            }
            case DataType::JSON: {
                getters.insert(std::make_pair(name, std::move(GenGetter<JSONFieldData>(field))));
                break;
            }
            case DataType::ARRAY: {
                switch (field->ElementType()) {
                    case DataType::BOOL: {
                        getters.insert(std::make_pair(name, std::move(GenGetter<ArrayBoolFieldData>(field))));
                        break;
                    }
                    case DataType::INT8: {
                        getters.insert(std::make_pair(name, std::move(GenGetter<ArrayInt8FieldData>(field))));
                        break;
                    }
                    case DataType::INT16: {
                        getters.insert(std::make_pair(name, std::move(GenGetter<ArrayInt16FieldData>(field))));
                        break;
                    }
                    case DataType::INT32: {
                        getters.insert(std::make_pair(name, std::move(GenGetter<ArrayInt32FieldData>(field))));
                        break;
                    }
                    case DataType::INT64: {
                        getters.insert(std::make_pair(name, std::move(GenGetter<ArrayInt64FieldData>(field))));
                        break;
                    }
                    case DataType::FLOAT: {
                        getters.insert(std::make_pair(name, std::move(GenGetter<ArrayFloatFieldData>(field))));
                        break;
                    }
                    case DataType::DOUBLE: {
                        getters.insert(std::make_pair(name, std::move(GenGetter<ArrayDoubleFieldData>(field))));
                        break;
                    }
                    case DataType::VARCHAR:
                    case DataType::GEOMETRY:
                    case DataType::TIMESTAMPTZ: {
                        getters.insert(std::make_pair(name, std::move(GenGetter<ArrayVarCharFieldData>(field))));
                        break;
                    }
                    case DataType::STRUCT: {
                        getters.insert(std::make_pair(name, std::move(GenGetter<StructFieldData>(field))));
                        break;
                    }
                    default:
                        // no need to return error here, for new unknown dat type, the data is not displayed,
                        // SearchResults::OutputFields/QueryResults::OutputFields can handle unknown dat types.
                        break;
                }
                break;
            }
            case DataType::BINARY_VECTOR: {
                getters.insert(std::make_pair(name, std::move(GenGetter<BinaryVecFieldData>(field))));
                break;
            }
            case DataType::FLOAT_VECTOR: {
                getters.insert(std::make_pair(name, std::move(GenGetter<FloatVecFieldData>(field))));
                break;
            }
            case DataType::FLOAT16_VECTOR: {
                getters.insert(std::make_pair(name, std::move(GenGetter<Float16VecFieldData>(field))));
                break;
            }
            case DataType::BFLOAT16_VECTOR: {
                getters.insert(std::make_pair(name, std::move(GenGetter<BFloat16VecFieldData>(field))));
                break;
            }
            case DataType::SPARSE_FLOAT_VECTOR: {
                getters.insert(std::make_pair(name, std::move(GenGetter<SparseFloatVecFieldData>(field))));
                break;
            }
            case DataType::INT8_VECTOR: {
                getters.insert(std::make_pair(name, std::move(GenGetter<Int8VecFieldData>(field))));
                break;
            }
            default:
                // no need to return error here, for new unknown dat type, the data is not displayed,
                // SearchResults::OutputFields/QueryResults::OutputFields can handle unknown dat types.
                break;
        }
    }
    return std::move(getters);
}

Status
GetRowCountOfFields(const std::vector<FieldDataPtr>& fields, size_t& count) {
    size_t first_cnt = 0;
    for (const auto& field : fields) {
        if (field != nullptr) {
            first_cnt = field->Count();
            break;
        }
    }
    for (const auto& field : fields) {
        if (field != nullptr && first_cnt != field->Count()) {
            return {StatusCode::INVALID_AGUMENT, "Row numbers of fields are not equal"};
        }
    }
    count = first_cnt;
    return Status::OK();
}

void
SetOutputRow(std::map<std::string, std::function<nlohmann::json(size_t)>>& getters, size_t i,
             const std::set<std::string>& output_names, EntityRow& row) {
    for (auto& getter : getters) {
        if (getter.first == DYNAMIC_FIELD) {
            // dynamic field special name "$meta", the value is a JSON dict
            // the server returns entire value of "$meta", we only pick the keys in output_names into the row
            // if the output_names contains DYNAMIC_FIELD, that means all dynamic fields need to be output
            auto meta = getter.second(i);
            if (output_names.find(DYNAMIC_FIELD) != output_names.end()) {
                for (auto& pair : meta.items()) {
                    row[pair.key()] = pair.value();
                }
            } else {
                for (auto& pair : meta.items()) {
                    if (output_names.find(pair.key()) != output_names.end()) {
                        row[pair.key()] = pair.value();
                    }
                }
            }
        } else {
            // non-dynamic fields
            row[getter.first] = getter.second(i);
        }
    }
}

Status
GetRowsFromFieldsData(const std::vector<FieldDataPtr>& fields, const std::set<std::string>& output_names,
                      EntityRows& rows) {
    rows.clear();
    size_t count = 0;
    auto status = GetRowCountOfFields(fields, count);
    if (!status.IsOk()) {
        return status;
    }

    auto getters = GenGetters(fields);
    for (auto i = 0; i < count; i++) {
        EntityRow row;
        SetOutputRow(getters, i, output_names, row);
        rows.emplace_back(std::move(row));
    }
    return Status::OK();
}

Status
GetRowFromFieldsData(const std::vector<FieldDataPtr>& fields, size_t i, const std::set<std::string>& output_names,
                     EntityRow& row) {
    row.clear();
    size_t count = 0;
    auto status = GetRowCountOfFields(fields, count);
    if (!status.IsOk()) {
        return status;
    }

    if (i >= count) {
        return {StatusCode::INVALID_AGUMENT, std::to_string(i) + " is out of bound: " + std::to_string(count)};
    }

    auto getters = GenGetters(fields);
    SetOutputRow(getters, i, output_names, row);
    return Status::OK();
}

uint64_t
DeduceGuaranteeTimestamp(const ConsistencyLevel& level, const std::string& db_name,
                         const std::string& collection_name) {
    if (level == ConsistencyLevel::NONE) {
        uint64_t ts = 1;
        return GtsDict::GetInstance().GetCollectionTs(db_name, collection_name, ts) ? ts : 1;
    }

    switch (level) {
        case ConsistencyLevel::STRONG:
            return 0;
        case ConsistencyLevel::SESSION: {
            uint64_t ts = 1;
            return GtsDict::GetInstance().GetCollectionTs(db_name, collection_name, ts) ? ts : 1;
        }
        case ConsistencyLevel::BOUNDED:
            return 2;  // let server side to determine the bounded time
        default:
            return 1;  // EVENTUALLY and others
    }
}

Status
DeduceTemplateArray(const nlohmann::json& array, proto::schema::TemplateArrayValue& rpc_array) {
    if (array.empty()) {
        return Status::OK();
    }
    const auto& first_ele = array.at(0);
    if (first_ele.is_boolean()) {
        for (const auto& ele : array) {
            if (!ele.is_boolean()) {
                return {StatusCode::INVALID_AGUMENT,
                        "Filter expression template is a list, the first value is Boolean, but some elements are not "
                        "Boolean"};
            }
            rpc_array.mutable_bool_data()->add_data(ele.get<bool>());
        }
    } else if (first_ele.is_number_integer()) {
        for (const auto& ele : array) {
            if (!ele.is_number_integer()) {
                return {StatusCode::INVALID_AGUMENT,
                        "Filter expression template is a list, the first value is Integer, but some elements are not "
                        "Integer"};
            }
            rpc_array.mutable_long_data()->add_data(ele.get<int64_t>());
        }
    } else if (first_ele.is_number_float()) {
        for (const auto& ele : array) {
            if (!ele.is_number_float()) {
                return {StatusCode::INVALID_AGUMENT,
                        "Filter expression template is a list, the first value is Double, but some elements are not "
                        "Double"};
            }
            rpc_array.mutable_double_data()->add_data(ele.get<double>());
        }
    } else if (first_ele.is_string()) {
        for (const auto& ele : array) {
            if (!ele.is_string()) {
                return {StatusCode::INVALID_AGUMENT,
                        "Filter expression template is a list, the first value is String, but some elements are not "
                        "String"};
            }
            rpc_array.mutable_string_data()->add_data(ele.get<std::string>());
        }
    } else if (first_ele.is_array()) {
        auto rpc_array_array = rpc_array.mutable_array_data()->add_data();
        for (const auto& ele : array) {
            if (!ele.is_array()) {
                return {
                    StatusCode::INVALID_AGUMENT,
                    "Filter expression template is a list, the first value is List, but some elements are not List"};
            }

            auto sub_array = rpc_array_array->mutable_array_data()->add_data();
            auto status = DeduceTemplateArray(ele, *sub_array);
            if (!status.IsOk()) {
                return status;
            }
        }
    }

    return Status::OK();
}

Status
ConvertFilterTemplates(const std::unordered_map<std::string, nlohmann::json>& templates,
                       ::google::protobuf::Map<std::string, proto::schema::TemplateValue>* rpc_templates) {
    for (const auto& pair : templates) {
        proto::schema::TemplateValue value;
        const auto& temp = pair.second;
        if (temp.is_array()) {
            auto array = value.mutable_array_val();
            auto status = DeduceTemplateArray(temp, *array);
            if (!status.IsOk()) {
                return status;
            }
        } else if (temp.is_boolean()) {
            value.set_bool_val(temp.get<bool>());
        } else if (temp.is_number_integer()) {
            value.set_int64_val(temp.get<int64_t>());
        } else if (temp.is_number_float()) {
            value.set_float_val(temp.get<double>());
        } else if (temp.is_string()) {
            value.set_string_val(temp.get<std::string>());
        } else {
            return {StatusCode::INVALID_AGUMENT, "Unsupported template type"};
        }
        rpc_templates->insert(std::make_pair(pair.first, value));
    }

    return Status::OK();
}

// current_db is the actual target db that the request is performed, for setting the GuaranteeTimestamp
// to compatible with old versions.
// for examples:
// - the MilvusClient connects to "my_db", the request.DatabaseName() is empty, target db is "my_db"
// - the MilvusClient connects to "", the request.DatabaseName() is empty, target db is "default"
// - the MilvusClient connects to "", the request.DatabaseName() is "my_db", target db is "my_db"
// - the MilvusClient connects to "db_1", the request.DatabaseName() is "db_2", target db is "db_2"
template <typename T>
Status
ConvertQueryRequest(const T& request, const std::string& current_db, proto::milvus::QueryRequest& rpc_request) {
    auto db_name = request.DatabaseName();
    if (!db_name.empty()) {
        rpc_request.set_db_name(db_name);
    }
    rpc_request.set_collection_name(request.CollectionName());
    for (const auto& partition_name : request.PartitionNames()) {
        rpc_request.add_partition_names(partition_name);
    }

    rpc_request.set_expr(request.Filter());
    if (!request.Filter().empty()) {
        auto rpc_templates = rpc_request.mutable_expr_template_values();
        const auto& templates = request.FilterTemplates();
        auto status = ConvertFilterTemplates(templates, rpc_templates);
        if (!status.IsOk()) {
            return status;
        }
    }

    for (const auto& field : request.OutputFields()) {
        rpc_request.add_output_fields(field);
    }

    // limit/offet etc.
    auto& params = request.ExtraParams();
    for (auto& pair : params) {
        auto kv_pair = rpc_request.add_query_params();
        kv_pair->set_key(pair.first);
        kv_pair->set_value(pair.second);
    }

    ConsistencyLevel level = request.GetConsistencyLevel();
    uint64_t guarantee_ts = DeduceGuaranteeTimestamp(level, current_db, request.CollectionName());
    rpc_request.set_guarantee_timestamp(guarantee_ts);

    if (level == ConsistencyLevel::NONE) {
        rpc_request.set_use_default_consistency(true);
    } else {
        rpc_request.set_consistency_level(ConsistencyLevelCast(level));
    }
    return Status::OK();
}

Status
ConvertQueryResults(const proto::milvus::QueryResults& rpc_results, QueryResults& results) {
    std::vector<milvus::FieldDataPtr> return_fields{};
    return_fields.reserve(rpc_results.fields_data_size());
    for (const auto& field_data : rpc_results.fields_data()) {
        FieldDataPtr field_ptr;
        auto status = CreateMilvusFieldData(field_data, field_ptr);
        if (!status.IsOk()) {
            return status;
        }
        return_fields.emplace_back(std::move(field_ptr));
    }

    std::set<std::string> output_names;
    for (const auto& name : rpc_results.output_fields()) {
        output_names.insert(name);
    }

    results = QueryResults(std::move(return_fields), output_names);
    return Status::OK();
}

// current_db is the actual target db that the request is performed, for setting the GuaranteeTimestamp
// to compatible with old versions.
// for examples:
// - the MilvusClient connects to "my_db", the request.DatabaseName() is empty, target db is "my_db"
// - the MilvusClient connects to "", the request.DatabaseName() is empty, target db is "default"
// - the MilvusClient connects to "", the request.DatabaseName() is "my_db", target db is "my_db"
// - the MilvusClient connects to "db_1", the request.DatabaseName() is "db_2", target db is "db_2"
template <typename T>
Status
ConvertSearchRequest(const T& request, const std::string& current_db, proto::milvus::SearchRequest& rpc_request) {
    if (!current_db.empty()) {
        rpc_request.set_db_name(current_db);
    }
    rpc_request.set_collection_name(request.CollectionName());
    rpc_request.set_dsl_type(proto::common::DslType::BoolExprV1);
    if (!request.Filter().empty()) {
        rpc_request.set_dsl(request.Filter());

        auto rpc_templates = rpc_request.mutable_expr_template_values();
        const auto& templates = request.FilterTemplates();
        auto status = ConvertFilterTemplates(templates, rpc_templates);
        if (!status.IsOk()) {
            return status;
        }
    }
    for (const auto& partition_name : request.PartitionNames()) {
        rpc_request.add_partition_names(partition_name);
    }
    for (const auto& output_field : request.OutputFields()) {
        rpc_request.add_output_fields(output_field);
    }

    // set target vectors
    if (request.TargetVectors() != nullptr) {
        auto status = SetTargetVectors(request.TargetVectors(), &rpc_request);
        if (!status.IsOk()) {
            return status;
        }
    } else {
        auto status = SetEmbeddingLists(request.EmbeddingLists(), &rpc_request);
        if (!status.IsOk()) {
            return status;
        }
    }

    auto setParamFunc = [&rpc_request](const std::string& key, const std::string& value) {
        auto kv_pair = rpc_request.add_search_params();
        kv_pair->set_key(key);
        kv_pair->set_value(value);
    };

    // set anns field name, if the name is empty and the collection has only one vector field,
    // milvus server will use the vector field name as anns name. If the collection has multiple
    // vector fields, user needs to explicitly provide an anns field name.
    auto anns_field = request.AnnsField();
    if (!anns_field.empty()) {
        setParamFunc(milvus::ANNS_FIELD, anns_field);
    }

    // for history reason, query() requires "limit", search() requires "topk"
    setParamFunc(milvus::TOPK, std::to_string(request.Limit()));

    // set this value only when client specified, otherwise let server to get it from index parameters
    if (request.MetricType() != MetricType::DEFAULT) {
        setParamFunc(milvus::METRIC_TYPE, std::to_string(request.MetricType()));
    }

    // extra params offset/round_decimal/group_by/radius/range_filter/nprobe etc.
    SetExtraParams(request.ExtraParams(), rpc_request.mutable_search_params());

    // consistency level
    ConsistencyLevel level = request.GetConsistencyLevel();
    uint64_t guarantee_ts = DeduceGuaranteeTimestamp(level, current_db, request.CollectionName());
    rpc_request.set_guarantee_timestamp(guarantee_ts);

    if (level == ConsistencyLevel::NONE) {
        rpc_request.set_use_default_consistency(true);
    } else {
        rpc_request.set_consistency_level(ConsistencyLevelCast(level));
    }
    return Status::OK();
}

Status
ConvertSearchResults(const proto::milvus::SearchResults& rpc_results, const std::string& pk_name,
                     SearchResults& results) {
    const auto& result_data = rpc_results.results();
    const auto& ids = result_data.ids();
    const auto& fields_data = result_data.fields_data();
    std::set<std::string> output_names;
    for (const auto& name : result_data.output_fields()) {
        output_names.insert(name);
    }

    // in milvus version older than v2.4.20, the primary_field_name() is empty, we need to
    // get the primary key field name from collection schema
    // if no pk_name is inputed, use a hard-code name "pk"
    std::string real_pk_name = result_data.primary_field_name();
    real_pk_name = real_pk_name.empty() ? pk_name : real_pk_name;
    real_pk_name = real_pk_name.empty() ? "pk" : real_pk_name;

    auto num_of_queries = result_data.num_queries();
    std::vector<int> topks{};
    topks.reserve(result_data.topks_size());
    for (int i = 0; i < result_data.topks_size(); ++i) {
        topks.push_back(result_data.topks(i));
    }
    std::vector<SingleResult> single_results;
    single_results.reserve(num_of_queries);
    int offset{0};
    for (int i = 0; i < num_of_queries; ++i) {
        std::vector<FieldDataPtr> item_fields_data;
        item_fields_data.reserve(fields_data.size());
        auto item_topk = topks[i];
        std::set<std::string> field_names;
        for (const auto& field_data : fields_data) {
            FieldDataPtr field_ptr;
            auto status = CreateMilvusFieldData(field_data, offset, item_topk, field_ptr);
            if (!status.IsOk()) {
                return status;
            }
            item_fields_data.emplace_back(std::move(field_ptr));
            field_names.insert(field_data.field_name());
        }
        std::string score_name = SCORE;
        while (field_names.find(score_name) != field_names.end()) {
            score_name = "_" + score_name;
        }

        FieldDataPtr id_field = CreateIDField(real_pk_name, ids, offset, item_topk);
        FieldDataPtr score_field = CreateScoreField(score_name, result_data, offset, item_topk);
        item_fields_data.emplace_back(std::move(id_field));
        item_fields_data.emplace_back(std::move(score_field));

        // if the server return different lenth of ids, scores, this line will throw an exception
        // we never saw such bug, just keep a protection here in case if it happens.
        try {
            single_results.emplace_back(real_pk_name, score_name, std::move(item_fields_data), output_names);
        } catch (const std::exception& e) {
            return {StatusCode::UNKNOWN_ERROR, "Not able to parse search results, error: " + std::string(e.what())};
        }
        offset += item_topk;
    }

    results = SearchResults(std::move(single_results));
    return Status::OK();
}

// current_db is the actual target db that the request is performed, for setting the GuaranteeTimestamp
// to compatible with old versions.
// for examples:
// - the MilvusClient connects to "my_db", the request.DatabaseName() is empty, target db is "my_db"
// - the MilvusClient connects to "", the request.DatabaseName() is empty, target db is "default"
// - the MilvusClient connects to "", the request.DatabaseName() is "my_db", target db is "my_db"
// - the MilvusClient connects to "db_1", the request.DatabaseName() is "db_2", target db is "db_2"
template <typename T>
Status
ConvertHybridSearchRequest(const T& request, const std::string& current_db,
                           proto::milvus::HybridSearchRequest& rpc_request) {
    auto db_name = request.DatabaseName();
    if (!db_name.empty()) {
        rpc_request.set_db_name(db_name);
    }
    rpc_request.set_collection_name(request.CollectionName());

    for (const auto& partition_name : request.PartitionNames()) {
        rpc_request.add_partition_names(partition_name);
    }
    for (const auto& output_field : request.OutputFields()) {
        rpc_request.add_output_fields(output_field);
    }

    for (const auto& sub_request : request.SubRequests()) {
        auto search_req = rpc_request.add_requests();
        if (sub_request->TargetVectors() != nullptr) {
            auto status = SetTargetVectors(sub_request->TargetVectors(), search_req);
            if (!status.IsOk()) {
                return status;
            }
        } else {
            auto status = SetEmbeddingLists(sub_request->EmbeddingLists(), search_req);
            if (!status.IsOk()) {
                return status;
            }
        }

        // set filter expression
        search_req->set_dsl_type(proto::common::DslType::BoolExprV1);
        if (!sub_request->Filter().empty()) {
            search_req->set_dsl(sub_request->Filter());

            auto rpc_templates = search_req->mutable_expr_template_values();
            const auto& templates = sub_request->FilterTemplates();
            auto status = ConvertFilterTemplates(templates, rpc_templates);
            if (!status.IsOk()) {
                return status;
            }
        }

        // set anns field name, if the name is empty and the collection has only one vector field,
        // milvus server will use the vector field name as anns name. If the collection has multiple
        // vector fields, user needs to explicitly provide an anns field name.
        auto anns_field = sub_request->AnnsField();
        if (!anns_field.empty()) {
            auto kv_pair = search_req->add_search_params();
            kv_pair->set_key(milvus::ANNS_FIELD);
            kv_pair->set_value(anns_field);
        }

        // for history reason, query() requires "limit", search() requires "topk"
        {
            auto kv_pair = search_req->add_search_params();
            kv_pair->set_key(milvus::TOPK);
            kv_pair->set_value(std::to_string(sub_request->Limit()));
        }

        // set this value only when client specified, otherwise let server to get it from index parameters
        if (sub_request->MetricType() != MetricType::DEFAULT) {
            auto kv_pair = search_req->add_search_params();
            kv_pair->set_key(milvus::METRIC_TYPE);
            kv_pair->set_value(std::to_string(sub_request->MetricType()));
        }

        // extra params offet/radius/range_filter/nprobe etc.
        SetExtraParams(sub_request->ExtraParams(), search_req->mutable_search_params());
    }

    auto setParamFunc = [&rpc_request](const std::string& key, const std::string& value) {
        auto kv_pair = rpc_request.add_rank_params();
        kv_pair->set_key(key);
        kv_pair->set_value(value);
    };

    // hybrid search is new interface, requires "limit"
    setParamFunc(LIMIT, std::to_string(request.Limit()));

    // extra params offset/round_decimal/group_by etc.
    for (auto& pair : request.ExtraParams()) {
        auto kv_pair = rpc_request.add_rank_params();
        kv_pair->set_key(pair.first);
        kv_pair->set_value(pair.second);
    }

    // set rerank
    auto reranker = request.Rerank();
    for (auto& pair : reranker->Params()) {
        setParamFunc(pair.first, pair.second);
    }

    // consistancy level
    ConsistencyLevel level = request.GetConsistencyLevel();
    uint64_t guarantee_ts = DeduceGuaranteeTimestamp(level, current_db, request.CollectionName());
    rpc_request.set_guarantee_timestamp(guarantee_ts);

    if (level == ConsistencyLevel::NONE) {
        rpc_request.set_use_default_consistency(true);
    } else {
        rpc_request.set_consistency_level(ConsistencyLevelCast(level));
    }
    return Status::OK();
}

template <typename T>
Status
CopyFieldDataRange(const FieldDataPtr& src, uint64_t from, uint64_t to, FieldDataPtr& target) {
    if (from >= to) {
        return {StatusCode::INVALID_AGUMENT, "Illegal copy range"};
    }

    auto src_ptr = std::static_pointer_cast<T>(src);
    const auto& src_data = src_ptr->Data();
    if (from == 0 && to == src->Count()) {
        target = src;
    } else {
        std::vector<typename T::ElementT> target_data{};
        target_data.reserve(to - from);
        std::copy(src_data.begin() + from, src_data.begin() + to, std::back_inserter(target_data));
        target = std::make_shared<T>(src->Name(), std::move(target_data));
    }
    return Status::OK();
}

Status
CopyFieldData(const FieldDataPtr& src, uint64_t from, uint64_t to, FieldDataPtr& target) {
    if (src == nullptr) {
        return {StatusCode::INVALID_AGUMENT, "Source field data is null pointer"};
    }
    if (from >= to || from >= src->Count()) {
        return {StatusCode::INVALID_AGUMENT, "Invalid range to copy"};
    }
    if (to > src->Count()) {
        to = src->Count();
    }

    switch (src->Type()) {
        case DataType::BOOL: {
            return CopyFieldDataRange<BoolFieldData>(src, from, to, target);
        }
        case DataType::INT8: {
            return CopyFieldDataRange<Int8FieldData>(src, from, to, target);
        }
        case DataType::INT16: {
            return CopyFieldDataRange<Int16FieldData>(src, from, to, target);
        }
        case DataType::INT32: {
            return CopyFieldDataRange<Int32FieldData>(src, from, to, target);
        }
        case DataType::INT64: {
            return CopyFieldDataRange<Int64FieldData>(src, from, to, target);
        }
        case DataType::FLOAT: {
            return CopyFieldDataRange<FloatFieldData>(src, from, to, target);
        }
        case DataType::DOUBLE: {
            return CopyFieldDataRange<DoubleFieldData>(src, from, to, target);
        }
        case DataType::VARCHAR:
        case DataType::GEOMETRY:
        case DataType::TIMESTAMPTZ: {
            return CopyFieldDataRange<VarCharFieldData>(src, from, to, target);
        }
        case DataType::JSON: {
            return CopyFieldDataRange<JSONFieldData>(src, from, to, target);
        }
        case DataType::ARRAY: {
            switch (src->ElementType()) {
                case DataType::BOOL: {
                    return CopyFieldDataRange<ArrayBoolFieldData>(src, from, to, target);
                }
                case DataType::INT8: {
                    return CopyFieldDataRange<ArrayInt8FieldData>(src, from, to, target);
                }
                case DataType::INT16: {
                    return CopyFieldDataRange<ArrayInt16FieldData>(src, from, to, target);
                }
                case DataType::INT32: {
                    return CopyFieldDataRange<ArrayInt32FieldData>(src, from, to, target);
                }
                case DataType::INT64: {
                    return CopyFieldDataRange<ArrayInt64FieldData>(src, from, to, target);
                }
                case DataType::FLOAT: {
                    return CopyFieldDataRange<ArrayFloatFieldData>(src, from, to, target);
                }
                case DataType::DOUBLE: {
                    return CopyFieldDataRange<ArrayDoubleFieldData>(src, from, to, target);
                }
                case DataType::VARCHAR:
                case DataType::GEOMETRY:
                case DataType::TIMESTAMPTZ: {
                    return CopyFieldDataRange<ArrayVarCharFieldData>(src, from, to, target);
                }
                case DataType::STRUCT: {
                    return CopyFieldDataRange<StructFieldData>(src, from, to, target);
                }
                default: {
                    std::string msg = "Unsupported element type: " + std::to_string(src->ElementType());
                    return {StatusCode::NOT_SUPPORTED, msg};
                }
            }
        }
        case DataType::BINARY_VECTOR: {
            return CopyFieldDataRange<BinaryVecFieldData>(src, from, to, target);
        }
        case DataType::FLOAT_VECTOR: {
            return CopyFieldDataRange<FloatVecFieldData>(src, from, to, target);
        }
        case DataType::FLOAT16_VECTOR: {
            return CopyFieldDataRange<Float16VecFieldData>(src, from, to, target);
        }
        case DataType::BFLOAT16_VECTOR: {
            return CopyFieldDataRange<BFloat16VecFieldData>(src, from, to, target);
        }
        case DataType::SPARSE_FLOAT_VECTOR: {
            return CopyFieldDataRange<SparseFloatVecFieldData>(src, from, to, target);
        }
        case DataType::INT8_VECTOR: {
            return CopyFieldDataRange<Int8VecFieldData>(src, from, to, target);
        }
        default: {
            return {StatusCode::NOT_SUPPORTED, "Unsupported field type: " + std::to_string(src->Type())};
        }
    }

    return Status::OK();
}

Status
CopyFieldsData(const std::vector<FieldDataPtr>& src, uint64_t from, uint64_t to, std::vector<FieldDataPtr>& target) {
    target.clear();
    for (const auto& field : src) {
        FieldDataPtr new_field;
        auto status = CopyFieldData(field, from, to, new_field);
        if (!status.IsOk()) {
            return status;
        }
        target.emplace_back(std::move(new_field));
    }
    return Status::OK();
}

template <typename T>
Status
Append(const FieldDataPtr& src, FieldDataPtr& target) {
    auto src_ptr = std::static_pointer_cast<T>(src);
    auto target_ptr = std::static_pointer_cast<T>(target);
    const auto& src_data = src_ptr->Data();
    target_ptr->Append(src_data);
    return Status::OK();
}

Status
AppendFieldData(const FieldDataPtr& from, FieldDataPtr& to) {
    if (from == nullptr || to == nullptr) {
        return {StatusCode::INVALID_AGUMENT, "Field data is null pointer"};
    }
    if ((from->Type() != to->Type()) || (from->ElementType() != to->ElementType())) {
        return {StatusCode::INVALID_AGUMENT, "Not able to append data, type mismatch"};
    }
    switch (from->Type()) {
        case DataType::BOOL: {
            return Append<BoolFieldData>(from, to);
        }
        case DataType::INT8: {
            return Append<Int8FieldData>(from, to);
        }
        case DataType::INT16: {
            return Append<Int16FieldData>(from, to);
        }
        case DataType::INT32: {
            return Append<Int32FieldData>(from, to);
        }
        case DataType::INT64: {
            return Append<Int64FieldData>(from, to);
        }
        case DataType::FLOAT: {
            return Append<FloatFieldData>(from, to);
        }
        case DataType::DOUBLE: {
            return Append<DoubleFieldData>(from, to);
        }
        case DataType::VARCHAR:
        case DataType::GEOMETRY:
        case DataType::TIMESTAMPTZ: {
            return Append<VarCharFieldData>(from, to);
        }
        case DataType::JSON: {
            return Append<JSONFieldData>(from, to);
        }
        case DataType::ARRAY: {
            switch (from->ElementType()) {
                case DataType::BOOL: {
                    return Append<ArrayBoolFieldData>(from, to);
                }
                case DataType::INT8: {
                    return Append<ArrayInt8FieldData>(from, to);
                }
                case DataType::INT16: {
                    return Append<ArrayInt16FieldData>(from, to);
                }
                case DataType::INT32: {
                    return Append<ArrayInt32FieldData>(from, to);
                }
                case DataType::INT64: {
                    return Append<ArrayInt64FieldData>(from, to);
                }
                case DataType::FLOAT: {
                    return Append<ArrayFloatFieldData>(from, to);
                }
                case DataType::DOUBLE: {
                    return Append<ArrayDoubleFieldData>(from, to);
                }
                case DataType::VARCHAR:
                case DataType::GEOMETRY:
                case DataType::TIMESTAMPTZ: {
                    return Append<ArrayVarCharFieldData>(from, to);
                }
                case DataType::STRUCT: {
                    return Append<StructFieldData>(from, to);
                }
                default: {
                    std::string msg = "Unsupported element type: " + std::to_string(from->ElementType());
                    return {StatusCode::NOT_SUPPORTED, msg};
                }
            }
        }
        case DataType::BINARY_VECTOR: {
            return Append<BinaryVecFieldData>(from, to);
        }
        case DataType::FLOAT_VECTOR: {
            return Append<FloatVecFieldData>(from, to);
        }
        case DataType::FLOAT16_VECTOR: {
            return Append<Float16VecFieldData>(from, to);
        }
        case DataType::BFLOAT16_VECTOR: {
            return Append<BFloat16VecFieldData>(from, to);
        }
        case DataType::SPARSE_FLOAT_VECTOR: {
            return Append<SparseFloatVecFieldData>(from, to);
        }
        case DataType::INT8_VECTOR: {
            return Append<Int8VecFieldData>(from, to);
        }
        default: {
            return {StatusCode::NOT_SUPPORTED, "Unsupported field type: " + std::to_string(from->Type())};
        }
    }
    return Status::OK();
}

Status
AppendSearchResult(const SingleResult& from, SingleResult& to) {
    if (to.GetRowCount() == 0) {
        // target is empty, no need to copy, just return the souce
        to = SingleResult(from);
        return Status::OK();
    }

    const auto& from_fields = from.OutputFields();
    for (const auto& from_field : from_fields) {
        auto to_field = to.OutputField(from_field->Name());
        if (to_field == nullptr) {
            // if the target doesn't have this field, skip without error in case if collection schema
            // have been changed during iteration
            continue;
        }
        auto status = AppendFieldData(from_field, to_field);
        if (!status.IsOk()) {
            return status;
        }
    }
    return Status::OK();
}

Status
IsAmbiguousParam(const std::string& key) {
    static std::set<std::string> s_ambiguous = {PARAMS, TOPK, ANNS_FIELD, METRIC_TYPE};
    if (s_ambiguous.find(key) != s_ambiguous.end()) {
        return Status{StatusCode::INVALID_AGUMENT,
                      "Ambiguous parameter: not allow to set '" + key + "' in extra params"};
    }
    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// explicitly instantiation of template methods to avoid link error
// query
template Status
ConvertQueryRequest<QueryIteratorArguments>(const QueryIteratorArguments&, const std::string&,
                                            proto::milvus::QueryRequest&);

template Status
ConvertQueryRequest<QueryArguments>(const QueryArguments&, const std::string&, proto::milvus::QueryRequest&);

template Status
ConvertQueryRequest<QueryIteratorRequest>(const QueryIteratorRequest&, const std::string&,
                                          proto::milvus::QueryRequest&);

template Status
ConvertQueryRequest<QueryRequest>(const QueryRequest&, const std::string&, proto::milvus::QueryRequest&);

// search
template Status
ConvertSearchRequest<SearchIteratorArguments>(const SearchIteratorArguments&, const std::string&,
                                              proto::milvus::SearchRequest&);

template Status
ConvertSearchRequest<SearchArguments>(const SearchArguments&, const std::string&, proto::milvus::SearchRequest&);

template Status
ConvertSearchRequest<SearchIteratorRequest>(const SearchIteratorRequest&, const std::string&,
                                            proto::milvus::SearchRequest&);

template Status
ConvertSearchRequest<SearchRequest>(const SearchRequest&, const std::string&, proto::milvus::SearchRequest&);

// hybrid search
template Status
ConvertHybridSearchRequest<HybridSearchArguments>(const HybridSearchArguments&, const std::string&,
                                                  proto::milvus::HybridSearchRequest&);

template Status
ConvertHybridSearchRequest<HybridSearchRequest>(const HybridSearchRequest&, const std::string&,
                                                proto::milvus::HybridSearchRequest&);

}  // namespace milvus
