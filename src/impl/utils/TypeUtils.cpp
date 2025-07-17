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

#include "TypeUtils.h"

namespace milvus {

proto::schema::DataType
DataTypeCast(DataType type) {
    switch (type) {
        case DataType::BOOL:
            return proto::schema::DataType::Bool;
        case DataType::INT8:
            return proto::schema::DataType::Int8;
        case DataType::INT16:
            return proto::schema::DataType::Int16;
        case DataType::INT32:
            return proto::schema::DataType::Int32;
        case DataType::INT64:
            return proto::schema::DataType::Int64;
        case DataType::FLOAT:
            return proto::schema::DataType::Float;
        case DataType::DOUBLE:
            return proto::schema::DataType::Double;
        case DataType::VARCHAR:
            return proto::schema::DataType::VarChar;
        case DataType::JSON:
            return proto::schema::DataType::JSON;
        case DataType::ARRAY:
            return proto::schema::DataType::Array;
        case DataType::BINARY_VECTOR:
            return proto::schema::DataType::BinaryVector;
        case DataType::FLOAT_VECTOR:
            return proto::schema::DataType::FloatVector;
        case DataType::SPARSE_FLOAT_VECTOR:
            return proto::schema::DataType::SparseFloatVector;
        default:
            return proto::schema::DataType::None;
    }
}

DataType
DataTypeCast(proto::schema::DataType type) {
    switch (type) {
        case proto::schema::DataType::Bool:
            return DataType::BOOL;
        case proto::schema::DataType::Int8:
            return DataType::INT8;
        case proto::schema::DataType::Int16:
            return DataType::INT16;
        case proto::schema::DataType::Int32:
            return DataType::INT32;
        case proto::schema::DataType::Int64:
            return DataType::INT64;
        case proto::schema::DataType::Float:
            return DataType::FLOAT;
        case proto::schema::DataType::Double:
            return DataType::DOUBLE;
        case proto::schema::DataType::VarChar:
            return DataType::VARCHAR;
        case proto::schema::DataType::JSON:
            return DataType::JSON;
        case proto::schema::DataType::Array:
            return DataType::ARRAY;
        case proto::schema::DataType::BinaryVector:
            return DataType::BINARY_VECTOR;
        case proto::schema::DataType::FloatVector:
            return DataType::FLOAT_VECTOR;
        case proto::schema::DataType::SparseFloatVector:
            return DataType::SPARSE_FLOAT_VECTOR;
        default:
            return DataType::UNKNOWN;
    }
}

MetricType
MetricTypeCast(const std::string& type) {
    if (type == "L2") {
        return MetricType::L2;
    }
    if (type == "IP") {
        return MetricType::IP;
    }
    if (type == "COSINE") {
        return MetricType::COSINE;
    }
    if (type == "HAMMING") {
        return MetricType::HAMMING;
    }
    if (type == "JACCARD") {
        return MetricType::JACCARD;
    }
    return MetricType::INVALID;
}

IndexType
IndexTypeCast(const std::string& type) {
    if (type == "FLAT") {
        return IndexType::FLAT;
    }
    if (type == "IVF_FLAT") {
        return IndexType::IVF_FLAT;
    }
    if (type == "IVF_SQ8") {
        return IndexType::IVF_SQ8;
    }
    if (type == "IVF_PQ") {
        return IndexType::IVF_PQ;
    }
    if (type == "HNSW") {
        return IndexType::HNSW;
    }
    if (type == "DISKANN") {
        return IndexType::DISKANN;
    }
    if (type == "AUTOINDEX") {
        return IndexType::AUTOINDEX;
    }
    if (type == "SCANN") {
        return IndexType::SCANN;
    }
    if (type == "GPU_IVF_FLAT") {
        return IndexType::GPU_IVF_FLAT;
    }
    if (type == "GPU_IVF_PQ") {
        return IndexType::GPU_IVF_PQ;
    }
    if (type == "GPU_BRUTE_FORCE") {
        return IndexType::GPU_BRUTE_FORCE;
    }
    if (type == "GPU_CAGRA") {
        return IndexType::GPU_CAGRA;
    }
    if (type == "BIN_FLAT") {
        return IndexType::BIN_FLAT;
    }
    if (type == "BIN_IVF_FLAT") {
        return IndexType::BIN_IVF_FLAT;
    }
    if (type == "Trie") {
        return IndexType::TRIE;
    }
    if (type == "STL_SORT") {
        return IndexType::STL_SORT;
    }
    if (type == "INVERTED") {
        return IndexType::INVERTED;
    }
    if (type == "SPARSE_INVERTED_INDEX") {
        return IndexType::SPARSE_INVERTED_INDEX;
    }
    if (type == "SPARSE_WAND") {
        return IndexType::SPARSE_WAND;
    }
    return IndexType::INVALID;
}

proto::schema::VectorField*
CreateProtoFieldData(const BinaryVecFieldData& field) {
    auto ret = new proto::schema::VectorField{};
    auto& data = field.Data();
    auto dim = data.front().size() * 8;
    auto& vectors_data = *(ret->mutable_binary_vector());
    vectors_data.reserve(data.size() * dim);
    for (const auto& item : data) {
        std::copy(item.begin(), item.end(), std::back_inserter(vectors_data));
    }
    ret->set_dim(static_cast<int>(dim));
    return ret;
}

proto::schema::VectorField*
CreateProtoFieldData(const FloatVecFieldData& field) {
    auto ret = new proto::schema::VectorField{};
    auto& data = field.Data();
    auto dim = data.front().size();
    auto& vectors_data = *(ret->mutable_float_vector()->mutable_data());
    vectors_data.Reserve(static_cast<int>(data.size() * dim));
    for (const auto& item : data) {
        vectors_data.Add(item.begin(), item.end());
    }
    ret->set_dim(static_cast<int>(dim));
    return ret;
}

std::string
EncodeSparseFloatVector(const SparseFloatVecFieldData::ElementT& sparse) {
    // Milvus server requires sparse vector to be transferred in little endian.
    // For each index-value pair, the first 4 bytes is a binary of unsigned int32,
    // the next 4 bytes is a binary of float32.
    // Each sparse is transfered with a binary of (8 * sparse.size()) bytes.
    std::vector<uint8_t> bytes(8 * sparse.size());
    int count = 0;
    for (const auto& pair : sparse) {
        int k = count * 8;
        uint32_t index = pair.first;
        bytes[k] = index & 0xFF;
        bytes[k + 1] = (index >> 8) & 0xFF;
        bytes[k + 2] = (index >> 16) & 0xFF;
        bytes[k + 3] = (index >> 24) & 0xFF;

        float value = pair.second;
        std::memcpy(&bytes[k + 4], &value, sizeof(float));
        count++;
    }

    return std::string{bytes.begin(), bytes.end()};
}

SparseFloatVecFieldData::ElementT
DecodeSparseFloatVector(std::string& bytes) {
    if (bytes.size() % 8 != 0) {
        throw std::runtime_error("Unexpected binary string is received from server side!");
    }

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

proto::schema::VectorField*
CreateProtoFieldData(const SparseFloatVecFieldData& field) {
    auto ret = new proto::schema::VectorField{};
    auto& data = field.Data();
    auto& vectors_data = *(ret->mutable_sparse_float_vector()->mutable_contents());
    vectors_data.Reserve(static_cast<int>(data.size()));
    size_t max_dim = 0;
    for (const auto& item : data) {
        vectors_data.Add(EncodeSparseFloatVector(item));
        max_dim = item.size() > max_dim ? item.size() : max_dim;
    }
    ret->set_dim(static_cast<int64_t>(max_dim));
    return ret;
}

proto::schema::ScalarField*
CreateProtoFieldData(const BoolFieldData& field) {
    auto ret = new proto::schema::ScalarField{};
    auto& data = field.Data();
    auto& scalars_data = *(ret->mutable_bool_data()->mutable_data());
    scalars_data.Add(data.begin(), data.end());
    return ret;
}

proto::schema::ScalarField*
CreateProtoFieldData(const Int8FieldData& field) {
    auto ret = new proto::schema::ScalarField{};
    auto& data = field.Data();
    auto& scalars_data = *(ret->mutable_int_data()->mutable_data());
    scalars_data.Add(data.begin(), data.end());
    return ret;
}

proto::schema::ScalarField*
CreateProtoFieldData(const Int16FieldData& field) {
    auto ret = new proto::schema::ScalarField{};
    auto& data = field.Data();
    auto& scalars_data = *(ret->mutable_int_data()->mutable_data());
    scalars_data.Add(data.begin(), data.end());
    return ret;
}

proto::schema::ScalarField*
CreateProtoFieldData(const Int32FieldData& field) {
    auto ret = new proto::schema::ScalarField{};
    auto& data = field.Data();
    auto& scalars_data = *(ret->mutable_int_data()->mutable_data());
    scalars_data.Add(data.begin(), data.end());
    return ret;
}

proto::schema::ScalarField*
CreateProtoFieldData(const Int64FieldData& field) {
    auto ret = new proto::schema::ScalarField{};
    auto& data = field.Data();
    auto& scalars_data = *(ret->mutable_long_data()->mutable_data());
    scalars_data.Add(data.begin(), data.end());
    return ret;
}

proto::schema::ScalarField*
CreateProtoFieldData(const FloatFieldData& field) {
    auto ret = new proto::schema::ScalarField{};
    auto& data = field.Data();
    auto& scalars_data = *(ret->mutable_float_data()->mutable_data());
    scalars_data.Add(data.begin(), data.end());
    return ret;
}

proto::schema::ScalarField*
CreateProtoFieldData(const DoubleFieldData& field) {
    auto ret = new proto::schema::ScalarField{};
    auto& data = field.Data();
    auto& scalars_data = *(ret->mutable_double_data()->mutable_data());
    scalars_data.Add(data.begin(), data.end());
    return ret;
}

proto::schema::ScalarField*
CreateProtoFieldData(const VarCharFieldData& field) {
    auto ret = new proto::schema::ScalarField{};
    auto& data = field.Data();
    auto& scalars_data = *(ret->mutable_string_data()->mutable_data());
    scalars_data.Add(data.begin(), data.end());
    return ret;
}

proto::schema::ScalarField*
CreateProtoFieldData(const JSONFieldData& field) {
    auto ret = new proto::schema::ScalarField{};
    auto& data = field.Data();
    auto& scalars_data = *(ret->mutable_json_data());
    for (const auto& item : data) {
        scalars_data.add_data(item.dump());
    }
    return ret;
}

proto::schema::ScalarField*
CreateProtoArrayFieldData(const Field& field) {
    auto ret = new proto::schema::ScalarField{};
    auto element_type = field.ElementType();
    auto& array_data = *(ret->mutable_array_data());
    array_data.set_element_type(DataTypeCast(element_type));

    switch (element_type) {
        case DataType::BOOL: {
            const auto& arrayField = dynamic_cast<const ArrayBoolFieldData&>(field);
            const auto& data = arrayField.Data();
            for (const auto& row : data) {
                auto& scalar_field = *(array_data.add_data());
                auto& scalars_data = *(scalar_field.mutable_bool_data()->mutable_data());
                scalars_data.Add(row.begin(), row.end());
            }
            break;
        }
        case DataType::INT8: {
            const auto& arrayField = dynamic_cast<const ArrayInt8FieldData&>(field);
            const auto& data = arrayField.Data();
            for (const auto& row : data) {
                auto& scalar_field = *(array_data.add_data());
                auto& scalars_data = *(scalar_field.mutable_int_data()->mutable_data());
                scalars_data.Add(row.begin(), row.end());
            }
            break;
        }
        case DataType::INT16: {
            const auto& arrayField = dynamic_cast<const ArrayInt16FieldData&>(field);
            const auto& data = arrayField.Data();
            for (const auto& row : data) {
                auto& scalar_field = *(array_data.add_data());
                auto& scalars_data = *(scalar_field.mutable_int_data()->mutable_data());
                scalars_data.Add(row.begin(), row.end());
            }
            break;
        }
        case DataType::INT32: {
            const auto& arrayField = dynamic_cast<const ArrayInt32FieldData&>(field);
            const auto& data = arrayField.Data();
            for (const auto& row : data) {
                auto& scalar_field = *(array_data.add_data());
                auto& scalars_data = *(scalar_field.mutable_int_data()->mutable_data());
                scalars_data.Add(row.begin(), row.end());
            }
            break;
        }
        case DataType::INT64: {
            const auto& arrayField = dynamic_cast<const ArrayInt64FieldData&>(field);
            const auto& data = arrayField.Data();
            for (const auto& row : data) {
                auto& scalar_field = *(array_data.add_data());
                auto& scalars_data = *(scalar_field.mutable_long_data()->mutable_data());
                scalars_data.Add(row.begin(), row.end());
            }
            break;
        }
        case DataType::FLOAT: {
            const auto& arrayField = dynamic_cast<const ArrayFloatFieldData&>(field);
            const auto& data = arrayField.Data();
            for (const auto& row : data) {
                auto& scalar_field = *(array_data.add_data());
                auto& scalars_data = *(scalar_field.mutable_float_data()->mutable_data());
                scalars_data.Add(row.begin(), row.end());
            }
            break;
        }
        case DataType::DOUBLE: {
            const auto& arrayField = dynamic_cast<const ArrayDoubleFieldData&>(field);
            const auto& data = arrayField.Data();
            for (const auto& row : data) {
                auto& scalar_field = *(array_data.add_data());
                auto& scalars_data = *(scalar_field.mutable_double_data()->mutable_data());
                scalars_data.Add(row.begin(), row.end());
            }
            break;
        }
        case DataType::VARCHAR: {
            const auto& arrayField = dynamic_cast<const ArrayVarCharFieldData&>(field);
            const auto& data = arrayField.Data();
            for (const auto& row : data) {
                auto& scalar_field = *(array_data.add_data());
                auto& scalars_data = *(scalar_field.mutable_string_data()->mutable_data());
                scalars_data.Add(row.begin(), row.end());
            }
            break;
        }
        default:
            // TODO: should throw error here
            break;
    }

    return ret;
}

proto::schema::FieldData
CreateProtoFieldData(const Field& field) {
    proto::schema::FieldData field_data;
    const auto field_type = field.Type();
    field_data.set_field_name(field.Name());
    field_data.set_type(DataTypeCast(field_type));

    switch (field_type) {
        case DataType::BINARY_VECTOR:
            field_data.set_allocated_vectors(CreateProtoFieldData(dynamic_cast<const BinaryVecFieldData&>(field)));
            break;
        case DataType::FLOAT_VECTOR:
            field_data.set_allocated_vectors(CreateProtoFieldData(dynamic_cast<const FloatVecFieldData&>(field)));
            break;
        case DataType::SPARSE_FLOAT_VECTOR:
            field_data.set_allocated_vectors(CreateProtoFieldData(dynamic_cast<const SparseFloatVecFieldData&>(field)));
            break;
        case DataType::BOOL:
            field_data.set_allocated_scalars(CreateProtoFieldData(dynamic_cast<const BoolFieldData&>(field)));
            break;
        case DataType::INT8:
            field_data.set_allocated_scalars(CreateProtoFieldData(dynamic_cast<const Int8FieldData&>(field)));
            break;
        case DataType::INT16:
            field_data.set_allocated_scalars(CreateProtoFieldData(dynamic_cast<const Int16FieldData&>(field)));
            break;
        case DataType::INT32:
            field_data.set_allocated_scalars(CreateProtoFieldData(dynamic_cast<const Int32FieldData&>(field)));
            break;
        case DataType::INT64:
            field_data.set_allocated_scalars(CreateProtoFieldData(dynamic_cast<const Int64FieldData&>(field)));
            break;
        case DataType::FLOAT:
            field_data.set_allocated_scalars(CreateProtoFieldData(dynamic_cast<const FloatFieldData&>(field)));
            break;
        case DataType::DOUBLE:
            field_data.set_allocated_scalars(CreateProtoFieldData(dynamic_cast<const DoubleFieldData&>(field)));
            break;
        case DataType::VARCHAR:
            field_data.set_allocated_scalars(CreateProtoFieldData(dynamic_cast<const VarCharFieldData&>(field)));
            break;
        case DataType::JSON:
            field_data.set_allocated_scalars(CreateProtoFieldData(dynamic_cast<const JSONFieldData&>(field)));
            break;
        case DataType::ARRAY:
            field_data.set_allocated_scalars(CreateProtoArrayFieldData(field));
            break;
        default:
            // TODO: should throw error here
            break;
    }

    return field_data;
}

FieldDataPtr
CreateMilvusArrayFieldData(const std::string& name, const milvus::proto::schema::ArrayArray& array_field, int offset,
                           int count) {
    const auto& scalars_data = array_field.data();
    auto begin = scalars_data.begin();
    if (offset < 0) {
        offset = 0;
    }
    std::advance(begin, offset);
    auto end = begin;
    // avoid overflow
    int arr_size = scalars_data.size();
    if (offset < arr_size) {
        if (offset + count > arr_size) {
            count = arr_size - offset;
        }
        if (count > 0) {
            std::advance(end, count);
        }
    }

    proto::schema::DataType element_type = array_field.element_type();
    switch (element_type) {
        case proto::schema::DataType::Bool: {
            std::vector<ArrayBoolFieldData::ElementT> arr;
            for (; begin != end; begin++) {
                arr.emplace_back(BuildFieldDataScalars<bool>((*begin).bool_data().data()));
            }
            return std::make_shared<ArrayBoolFieldData>(name, arr);
        }
        case proto::schema::DataType::Int8: {
            std::vector<ArrayInt8FieldData::ElementT> arr;
            for (; begin != end; begin++) {
                arr.emplace_back(BuildFieldDataScalars<int8_t>((*begin).int_data().data()));
            }
            return std::make_shared<ArrayInt8FieldData>(name, arr);
        }
        case proto::schema::DataType::Int16: {
            std::vector<ArrayInt16FieldData::ElementT> arr;
            for (; begin != end; begin++) {
                arr.emplace_back(BuildFieldDataScalars<int16_t>((*begin).int_data().data()));
            }
            return std::make_shared<ArrayInt16FieldData>(name, arr);
        }
        case proto::schema::DataType::Int32: {
            std::vector<ArrayInt32FieldData::ElementT> arr;
            for (; begin != end; begin++) {
                arr.emplace_back(BuildFieldDataScalars<int32_t>((*begin).int_data().data()));
            }
            return std::make_shared<ArrayInt32FieldData>(name, arr);
        }
        case proto::schema::DataType::Int64: {
            std::vector<ArrayInt64FieldData::ElementT> arr;
            for (; begin != end; begin++) {
                arr.emplace_back(BuildFieldDataScalars<int64_t>((*begin).long_data().data()));
            }
            return std::make_shared<ArrayInt64FieldData>(name, arr);
        }
        case proto::schema::DataType::Float: {
            std::vector<ArrayFloatFieldData::ElementT> arr;
            for (; begin != end; begin++) {
                arr.emplace_back(BuildFieldDataScalars<float>((*begin).float_data().data()));
            }
            return std::make_shared<ArrayFloatFieldData>(name, arr);
        }
        case proto::schema::DataType::Double: {
            std::vector<ArrayDoubleFieldData::ElementT> arr;
            for (; begin != end; begin++) {
                arr.emplace_back(BuildFieldDataScalars<double>((*begin).double_data().data()));
            }
            return std::make_shared<ArrayDoubleFieldData>(name, arr);
        }
        case proto::schema::DataType::VarChar: {
            std::vector<ArrayVarCharFieldData::ElementT> arr;
            for (; begin != end; begin++) {
                arr.emplace_back(BuildFieldDataScalars<std::string>((*begin).string_data().data()));
            }
            return std::make_shared<ArrayVarCharFieldData>(name, arr);
        }
        default:
            return nullptr;
    }
}

FieldDataPtr
CreateMilvusFieldData(const milvus::proto::schema::FieldData& field_data, size_t offset, size_t count) {
    auto field_type = field_data.type();
    const auto& name = field_data.field_name();

    switch (field_type) {
        case proto::schema::DataType::BinaryVector: {
            std::vector<BinaryVecFieldData::ElementT> vectors = BuildFieldDataVectors<std::string>(
                field_data.vectors().dim() / 8, field_data.vectors().binary_vector(), offset, count);
            return std::make_shared<BinaryVecFieldData>(name, std::move(vectors));
        }
        case proto::schema::DataType::FloatVector: {
            std::vector<FloatVecFieldData::ElementT> vectors = BuildFieldDataVectors<std::vector<float>>(
                field_data.vectors().dim(), field_data.vectors().float_vector().data(), offset, count);
            return std::make_shared<FloatVecFieldData>(name, std::move(vectors));
        }
        case proto::schema::DataType::SparseFloatVector: {
            std::vector<SparseFloatVecFieldData::ElementT> vectors =
                BuildFieldDataSparseVectors(field_data.vectors().sparse_float_vector().contents(), offset, count);
            return std::make_shared<SparseFloatVecFieldData>(name, std::move(vectors));
        }
        case proto::schema::DataType::Bool:
            return std::make_shared<BoolFieldData>(
                name, BuildFieldDataScalars<bool>(field_data.scalars().bool_data().data(), offset, count));

        case proto::schema::DataType::Int8:
            return std::make_shared<Int8FieldData>(
                name, BuildFieldDataScalars<int8_t>(field_data.scalars().int_data().data(), offset, count));

        case proto::schema::DataType::Int16:
            return std::make_shared<Int16FieldData>(
                name, BuildFieldDataScalars<int16_t>(field_data.scalars().int_data().data(), offset, count));

        case proto::schema::DataType::Int32:
            return std::make_shared<Int32FieldData>(
                name, BuildFieldDataScalars<int32_t>(field_data.scalars().int_data().data(), offset, count));

        case proto::schema::DataType::Int64:
            return std::make_shared<Int64FieldData>(
                name, BuildFieldDataScalars<int64_t>(field_data.scalars().long_data().data(), offset, count));

        case proto::schema::DataType::Float:
            return std::make_shared<FloatFieldData>(
                name, BuildFieldDataScalars<float>(field_data.scalars().float_data().data(), offset, count));

        case proto::schema::DataType::Double:
            return std::make_shared<DoubleFieldData>(
                name, BuildFieldDataScalars<double>(field_data.scalars().double_data().data(), offset, count));

        case proto::schema::DataType::VarChar:
            return std::make_shared<VarCharFieldData>(
                name, BuildFieldDataScalars<std::string>(field_data.scalars().string_data().data(), offset, count));

        case proto::schema::DataType::JSON: {
            std::vector<nlohmann::json> objects;
            const auto& scalars_data = field_data.scalars().json_data().data();
            for (const auto& s : scalars_data) {
                objects.emplace_back(nlohmann::json::parse(s));
            }
            return std::make_shared<JSONFieldData>(name, BuildFieldDataScalars<nlohmann::json>(objects, offset, count));
        }

        case proto::schema::DataType::Array: {
            return CreateMilvusArrayFieldData(name, field_data.scalars().array_data(), offset, count);
        }
        default:
            return nullptr;
    }
}

FieldDataPtr
CreateMilvusFieldData(const milvus::proto::schema::FieldData& field_data) {
    auto field_type = field_data.type();
    const auto& name = field_data.field_name();

    switch (field_type) {
        case proto::schema::DataType::BinaryVector:
            return std::make_shared<BinaryVecFieldData>(
                name, BuildFieldDataVectors<std::string>(field_data.vectors().dim() / 8,
                                                         field_data.vectors().binary_vector()));

        case proto::schema::DataType::FloatVector:
            return std::make_shared<FloatVecFieldData>(
                name, BuildFieldDataVectors<std::vector<float>>(field_data.vectors().dim(),
                                                                field_data.vectors().float_vector().data()));

        case proto::schema::DataType::SparseFloatVector: {
            auto content = field_data.vectors().sparse_float_vector().contents();
            return std::make_shared<SparseFloatVecFieldData>(name,
                                                             BuildFieldDataSparseVectors(content, 0, content.size()));
        }
        case proto::schema::DataType::Bool:
            return std::make_shared<BoolFieldData>(
                name, BuildFieldDataScalars<bool>(field_data.scalars().bool_data().data()));

        case proto::schema::DataType::Int8:
            return std::make_shared<Int8FieldData>(
                name, BuildFieldDataScalars<int8_t>(field_data.scalars().int_data().data()));

        case proto::schema::DataType::Int16:
            return std::make_shared<Int16FieldData>(
                name, BuildFieldDataScalars<int16_t>(field_data.scalars().int_data().data()));

        case proto::schema::DataType::Int32:
            return std::make_shared<Int32FieldData>(
                name, BuildFieldDataScalars<int32_t>(field_data.scalars().int_data().data()));

        case proto::schema::DataType::Int64:
            return std::make_shared<Int64FieldData>(
                name, BuildFieldDataScalars<int64_t>(field_data.scalars().long_data().data()));

        case proto::schema::DataType::Float:
            return std::make_shared<FloatFieldData>(
                name, BuildFieldDataScalars<float>(field_data.scalars().float_data().data()));

        case proto::schema::DataType::Double:
            return std::make_shared<DoubleFieldData>(
                name, BuildFieldDataScalars<double>(field_data.scalars().double_data().data()));

        case proto::schema::DataType::VarChar:
            return std::make_shared<VarCharFieldData>(
                name, BuildFieldDataScalars<std::string>(field_data.scalars().string_data().data()));

        case proto::schema::DataType::JSON: {
            std::vector<nlohmann::json> objects;
            const auto& scalars_data = field_data.scalars().json_data().data();
            for (const auto& s : scalars_data) {
                objects.emplace_back(nlohmann::json::parse(s));
            }
            return std::make_shared<JSONFieldData>(name, BuildFieldDataScalars<nlohmann::json>(objects));
        }

        case proto::schema::DataType::Array: {
            const auto& scalars_data = field_data.scalars().array_data();
            return CreateMilvusArrayFieldData(name, scalars_data, 0, scalars_data.data().size());
        }
        default:
            return nullptr;
    }
}

IDArray
CreateIDArray(const proto::schema::IDs& ids) {
    if (ids.has_int_id()) {
        std::vector<int64_t> int_array;
        auto& int_ids = ids.int_id();
        int_array.reserve(int_ids.data_size());
        std::copy(int_ids.data().begin(), int_ids.data().end(), std::back_inserter(int_array));
        return IDArray(int_array);
    } else {
        std::vector<std::string> str_array;
        auto& str_ids = ids.str_id();
        str_array.reserve(str_ids.data_size());
        std::copy(str_ids.data().begin(), str_ids.data().end(), std::back_inserter(str_array));
        return IDArray(str_array);
    }
}

IDArray
CreateIDArray(const proto::schema::IDs& ids, size_t offset, size_t size) {
    if (ids.has_int_id()) {
        std::vector<int64_t> int_array;
        auto& int_ids = ids.int_id();
        int_array.reserve(size);
        auto it = int_ids.data().begin();
        std::advance(it, offset);
        auto it_end = it;
        std::advance(it_end, size);
        std::copy(it, it_end, std::back_inserter(int_array));
        return IDArray(int_array);
    } else {
        std::vector<std::string> str_array;
        auto& str_ids = ids.str_id();
        str_array.reserve(size);
        auto it = str_ids.data().begin();
        std::advance(it, offset);
        auto it_end = it;
        std::advance(it_end, size);
        std::copy(it, it_end, std::back_inserter(str_array));
        return IDArray(str_array);
    }
}

void
ConvertFieldSchema(const proto::schema::FieldSchema& proto_schema, FieldSchema& field_schema) {
    field_schema.SetName(proto_schema.name());
    field_schema.SetDescription(proto_schema.description());
    field_schema.SetPrimaryKey(proto_schema.is_primary_key());
    field_schema.SetAutoID(proto_schema.autoid());
    field_schema.SetDataType(DataTypeCast(proto_schema.data_type()));

    std::map<std::string, std::string> params;
    for (int k = 0; k < proto_schema.type_params_size(); ++k) {
        auto& kv = proto_schema.type_params(k);
        params.emplace(kv.key(), kv.value());
    }
    field_schema.SetTypeParams(std::move(params));
}

void
ConvertCollectionSchema(const proto::schema::CollectionSchema& proto_schema, CollectionSchema& schema) {
    schema.SetName(proto_schema.name());
    schema.SetDescription(proto_schema.description());
    schema.SetEnableDynamicField(proto_schema.enable_dynamic_field());

    for (int i = 0; i < proto_schema.fields_size(); ++i) {
        auto& proto_field = proto_schema.fields(i);
        FieldSchema field_schema;
        ConvertFieldSchema(proto_field, field_schema);
        schema.AddField(std::move(field_schema));
    }
}

void
ConvertFieldSchema(const FieldSchema& schema, proto::schema::FieldSchema& proto_schema) {
    proto_schema.set_name(schema.Name());
    proto_schema.set_description(schema.Description());
    proto_schema.set_is_primary_key(schema.IsPrimaryKey());
    proto_schema.set_autoid(schema.AutoID());
    proto_schema.set_data_type(DataTypeCast(schema.FieldDataType()));

    for (auto& kv : schema.TypeParams()) {
        auto pair = proto_schema.add_type_params();
        pair->set_key(kv.first);
        pair->set_value(kv.second);
    }
}

void
ConvertCollectionSchema(const CollectionSchema& schema, proto::schema::CollectionSchema& proto_schema) {
    proto_schema.set_name(schema.Name());
    proto_schema.set_description(schema.Description());

    for (auto& field : schema.Fields()) {
        auto proto_field = proto_schema.add_fields();
        ConvertFieldSchema(field, *proto_field);
    }
}

SegmentState
SegmentStateCast(proto::common::SegmentState state) {
    switch (state) {
        case proto::common::SegmentState::Dropped:
            return SegmentState::DROPPED;
        case proto::common::SegmentState::Flushed:
            return SegmentState::FLUSHED;
        case proto::common::SegmentState::Flushing:
            return SegmentState::FLUSHING;
        case proto::common::SegmentState::Growing:
            return SegmentState::GROWING;
        case proto::common::SegmentState::NotExist:
            return SegmentState::NOT_EXIST;
        case proto::common::SegmentState::Sealed:
            return SegmentState::SEALED;
        default:
            return SegmentState::UNKNOWN;
    }
}

proto::common::SegmentState
SegmentStateCast(SegmentState state) {
    switch (state) {
        case SegmentState::DROPPED:
            return proto::common::SegmentState::Dropped;
        case SegmentState::FLUSHED:
            return proto::common::SegmentState::Flushed;
        case SegmentState::FLUSHING:
            return proto::common::SegmentState::Flushing;
        case SegmentState::GROWING:
            return proto::common::SegmentState::Growing;
        case SegmentState::NOT_EXIST:
            return proto::common::SegmentState::NotExist;
        case SegmentState::SEALED:
            return proto::common::SegmentState::Sealed;
        default:
            return proto::common::SegmentState::SegmentStateNone;
    }
}

IndexStateCode
IndexStateCast(proto::common::IndexState state) {
    switch (state) {
        case proto::common::IndexState::IndexStateNone:
            return IndexStateCode::NONE;
        case proto::common::IndexState::Unissued:
            return IndexStateCode::UNISSUED;
        case proto::common::IndexState::InProgress:
            return IndexStateCode::IN_PROGRESS;
        case proto::common::IndexState::Finished:
            return IndexStateCode::FINISHED;
        default:
            return IndexStateCode::FAILED;
    }
}

bool
IsVectorType(DataType type) {
    return (DataType::BINARY_VECTOR == type || DataType::FLOAT_VECTOR == type);
}

std::string
Base64Encode(const std::string& val) {
    const char* base64_chars = {
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
        "+/"};

    auto len = val.size();
    auto len_encoded = (len + 2) / 3 * 4;
    std::string ret;
    ret.reserve(len_encoded);

    size_t pos = 0;

    while (pos < len) {
        ret.push_back(base64_chars[(val[pos + 0] & 0xfc) >> 2]);

        if (pos + 1 < len) {
            ret.push_back(base64_chars[((val[pos + 0] & 0x03) << 4) + ((val[pos + 1] & 0xf0) >> 4)]);

            if (pos + 2 < len) {
                ret.push_back(base64_chars[((val[pos + 1] & 0x0f) << 2) + ((val[pos + 2] & 0xc0) >> 6)]);
                ret.push_back(base64_chars[val[pos + 2] & 0x3f]);
            } else {
                ret.push_back(base64_chars[(val[pos + 1] & 0x0f) << 2]);
                ret.push_back('=');
            }
        } else {
            ret.push_back(base64_chars[(val[pos + 0] & 0x03) << 4]);
            ret.push_back('=');
            ret.push_back('=');
        }

        pos += 3;
    }

    return ret;
}

proto::common::ConsistencyLevel
ConsistencyLevelCast(const ConsistencyLevel& level) {
    switch (level) {
        case ConsistencyLevel::STRONG:
            return proto::common::ConsistencyLevel::Strong;
        case ConsistencyLevel::SESSION:
            return proto::common::ConsistencyLevel::Session;
        case ConsistencyLevel::EVENTUALLY:
            return proto::common::ConsistencyLevel::Eventually;
        default:
            return proto::common::ConsistencyLevel::Bounded;
    }
}

ConsistencyLevel
ConsistencyLevelCast(const proto::common::ConsistencyLevel& level) {
    switch (level) {
        case proto::common::ConsistencyLevel::Strong:
            return ConsistencyLevel::STRONG;
        case proto::common::ConsistencyLevel::Session:
            return ConsistencyLevel::SESSION;
        case proto::common::ConsistencyLevel::Eventually:
            return ConsistencyLevel::EVENTUALLY;
        default:
            return ConsistencyLevel::BOUNDED;
    }
}

}  // namespace milvus

namespace std {
std::string
to_string(milvus::MetricType metric_type) {
    switch (metric_type) {
        case milvus::MetricType::L2:
            return "L2";
        case milvus::MetricType::IP:
            return "IP";
        case milvus::MetricType::COSINE:
            return "COSINE";
        case milvus::MetricType::HAMMING:
            return "HAMMING";
        case milvus::MetricType::JACCARD:
            return "JACCARD";
        case milvus::MetricType::INVALID:
        default:
            return "INVALID";
    }
}
std::string
to_string(milvus::IndexType index_type) {
    switch (index_type) {
        case milvus::IndexType::FLAT:
            return "FLAT";
        case milvus::IndexType::IVF_FLAT:
            return "IVF_FLAT";
        case milvus::IndexType::IVF_PQ:
            return "IVF_PQ";
        case milvus::IndexType::IVF_SQ8:
            return "IVF_SQ8";
        case milvus::IndexType::HNSW:
            return "HNSW";
        case milvus::IndexType::DISKANN:
            return "DISKANN";
        case milvus::IndexType::AUTOINDEX:
            return "AUTOINDEX";
        case milvus::IndexType::SCANN:
            return "SCANN";
        case milvus::IndexType::GPU_IVF_FLAT:
            return "GPU_IVF_FLAT";
        case milvus::IndexType::GPU_IVF_PQ:
            return "GPU_IVF_PQ";
        case milvus::IndexType::GPU_BRUTE_FORCE:
            return "GPU_BRUTE_FORCE";
        case milvus::IndexType::GPU_CAGRA:
            return "GPU_CAGRA";
        case milvus::IndexType::BIN_FLAT:
            return "BIN_FLAT";
        case milvus::IndexType::BIN_IVF_FLAT:
            return "BIN_IVF_FLAT";
        case milvus::IndexType::TRIE:
            return "Trie";
        case milvus::IndexType::STL_SORT:
            return "STL_SORT";
        case milvus::IndexType::INVERTED:
            return "INVERTED";
        case milvus::IndexType::SPARSE_INVERTED_INDEX:
            return "SPARSE_INVERTED_INDEX";
        case milvus::IndexType::SPARSE_WAND:
            return "SPARSE_WAND";
        default:
            return "INVALID";
    }
}

std::string
to_string(milvus::DataType data_type) {
    static const std::map<milvus::DataType, std::string> name_map = {
        {milvus::DataType::BOOL, "BOOL"},
        {milvus::DataType::INT8, "INT8"},
        {milvus::DataType::INT16, "INT8"},
        {milvus::DataType::INT32, "INT32"},
        {milvus::DataType::INT64, "INT64"},
        {milvus::DataType::FLOAT, "FLOAT"},
        {milvus::DataType::DOUBLE, "DOUBLE"},
        {milvus::DataType::VARCHAR, "VARCHAR"},
        {milvus::DataType::JSON, "JSON"},
        {milvus::DataType::ARRAY, "ARRAY"},
        {milvus::DataType::BINARY_VECTOR, "BINARY_VECTOR"},
        {milvus::DataType::FLOAT_VECTOR, "FLOAT_VECTOR"},
        {milvus::DataType::SPARSE_FLOAT_VECTOR, "SPARSE_FLOAT_VECTOR"},
    };
    auto it = name_map.find(data_type);
    if (it == name_map.end()) {
        return "Unknow DataType";
    }
    return it->second;
}

}  // namespace std
