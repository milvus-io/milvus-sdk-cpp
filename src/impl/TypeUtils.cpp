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

bool
operator==(const proto::schema::FieldData& lhs, const BoolFieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (not scalars.has_bool_data()) {
        return false;
    }
    const auto& scalars_data = scalars.bool_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

bool
operator==(const proto::schema::FieldData& lhs, const Int8FieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (not scalars.has_int_data()) {
        return false;
    }
    const auto& scalars_data = scalars.int_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

bool
operator==(const proto::schema::FieldData& lhs, const Int16FieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (not scalars.has_int_data()) {
        return false;
    }
    const auto& scalars_data = scalars.int_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

bool
operator==(const proto::schema::FieldData& lhs, const Int32FieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (not scalars.has_int_data()) {
        return false;
    }
    const auto& scalars_data = scalars.int_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

bool
operator==(const proto::schema::FieldData& lhs, const Int64FieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (not scalars.has_long_data()) {
        return false;
    }
    const auto& scalars_data = scalars.long_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

bool
operator==(const proto::schema::FieldData& lhs, const FloatFieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (not scalars.has_float_data()) {
        return false;
    }
    const auto& scalars_data = scalars.float_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

bool
operator==(const proto::schema::FieldData& lhs, const DoubleFieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (not scalars.has_double_data()) {
        return false;
    }
    const auto& scalars_data = scalars.double_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

bool
operator==(const proto::schema::FieldData& lhs, const StringFieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (not scalars.has_string_data()) {
        return false;
    }
    const auto& scalars_data = scalars.string_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

bool
operator==(const proto::schema::FieldData& lhs, const BinaryVecFieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_vectors()) {
        return false;
    }
    size_t dim = 0;
    if (rhs.Count() > 0) {
        dim = rhs.Data().front().size();
    }

    const auto& vectors = lhs.vectors();
    if (vectors.has_float_vector()) {
        return false;
    }
    const auto& vectors_data = vectors.binary_vector();
    if (vectors_data.size() != (rhs.Count() * dim)) {
        return false;
    }

    auto it = vectors_data.begin();
    for (const auto& item : rhs.Data()) {
        if (!std::equal(item.begin(), item.end(), it, [](uint8_t a, char b) { return static_cast<char>(a) == b; })) {
            return false;
        }
        std::advance(it, dim);
    }
    return true;
}

bool
operator==(const proto::schema::FieldData& lhs, const FloatVecFieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_vectors()) {
        return false;
    }
    size_t dim = 0;
    if (rhs.Count() > 0) {
        dim = rhs.Data().front().size();
    }

    const auto& vectors = lhs.vectors();
    if (!vectors.has_float_vector()) {
        return false;
    }
    const auto& vectors_data = vectors.float_vector().data();
    if (vectors_data.size() != (rhs.Count() * dim)) {
        return false;
    }

    auto it = vectors_data.begin();
    for (const auto& item : rhs.Data()) {
        if (!std::equal(item.begin(), item.end(), it)) {
            return false;
        }
        std::advance(it, dim);
    }
    return true;
}

bool
operator==(const proto::schema::FieldData& lhs, const Field& rhs) {
    auto data_type = rhs.Type();
    switch (data_type) {
        case DataType::BOOL:
            return lhs == dynamic_cast<const BoolFieldData&>(rhs);
        case DataType::INT8:
            return lhs == dynamic_cast<const Int8FieldData&>(rhs);
        case DataType::INT16:
            return lhs == dynamic_cast<const Int16FieldData&>(rhs);
        case DataType::INT32:
            return lhs == dynamic_cast<const Int32FieldData&>(rhs);
        case DataType::INT64:
            return lhs == dynamic_cast<const Int64FieldData&>(rhs);
        case DataType::FLOAT:
            return lhs == dynamic_cast<const FloatFieldData&>(rhs);
        case DataType::DOUBLE:
            return lhs == dynamic_cast<const DoubleFieldData&>(rhs);
        case DataType::STRING:
            return lhs == dynamic_cast<const StringFieldData&>(rhs);
        case DataType::BINARY_VECTOR:
            return lhs == dynamic_cast<const BinaryVecFieldData&>(rhs);
        case DataType::FLOAT_VECTOR:
            return lhs == dynamic_cast<const FloatVecFieldData&>(rhs);
        default:
            return false;
    }
}

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
        case DataType::STRING:
            return proto::schema::DataType::String;
        case DataType::BINARY_VECTOR:
            return proto::schema::DataType::BinaryVector;
        case DataType::FLOAT_VECTOR:
            return proto::schema::DataType::FloatVector;
        default:
            return proto::schema::DataType::None;
    }
}

proto::schema::VectorField*
CreateProtoFieldData(const BinaryVecFieldData& field) {
    auto ret = new proto::schema::VectorField{};
    auto& data = field.Data();
    auto dim = data.front().size();
    auto& vectors_data = *(ret->mutable_binary_vector());
    vectors_data.reserve(data.size() * dim);
    for (const auto& item : data) {
        std::copy(item.begin(), item.end(), std::back_inserter(vectors_data));
    }
    ret->set_dim(dim);
    return ret;
}

proto::schema::VectorField*
CreateProtoFieldData(const FloatVecFieldData& field) {
    auto ret = new proto::schema::VectorField{};
    auto& data = field.Data();
    auto dim = data.front().size();
    auto& vectors_data = *(ret->mutable_float_vector()->mutable_data());
    vectors_data.Reserve(data.size() * dim);
    for (const auto& item : data) {
        vectors_data.Add(item.begin(), item.end());
    }
    ret->set_dim(dim);
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
CreateProtoFieldData(const StringFieldData& field) {
    auto ret = new proto::schema::ScalarField{};
    auto& data = field.Data();
    auto& scalars_data = *(ret->mutable_string_data());
    for (const auto& item : data) {
        scalars_data.add_data(item);
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
        case DataType::BINARY_VECTOR: {
            const auto* original_field = dynamic_cast<const BinaryVecFieldData*>(&field);
            if (original_field) {
                field_data.set_allocated_vectors(CreateProtoFieldData(*original_field));
            }
        } break;

        case DataType::FLOAT_VECTOR: {
            const auto* original_field = dynamic_cast<const FloatVecFieldData*>(&field);
            if (original_field) {
                field_data.set_allocated_vectors(CreateProtoFieldData(*original_field));
            }
        } break;

        case DataType::BOOL: {
            const auto* original_field = dynamic_cast<const BoolFieldData*>(&field);
            if (original_field) {
                field_data.set_allocated_scalars(CreateProtoFieldData(*original_field));
            }
        } break;

        case DataType::INT8: {
            const auto* original_field = dynamic_cast<const Int8FieldData*>(&field);
            if (original_field) {
                field_data.set_allocated_scalars(CreateProtoFieldData(*original_field));
            }
        } break;
        case DataType::INT16: {
            const auto* original_field = dynamic_cast<const Int16FieldData*>(&field);
            if (original_field) {
                field_data.set_allocated_scalars(CreateProtoFieldData(*original_field));
            }
        } break;
        case DataType::INT32: {
            const auto* original_field = dynamic_cast<const Int32FieldData*>(&field);
            if (original_field) {
                field_data.set_allocated_scalars(CreateProtoFieldData(*original_field));
            }
        } break;
        case DataType::INT64: {
            const auto* original_field = dynamic_cast<const Int64FieldData*>(&field);
            if (original_field) {
                field_data.set_allocated_scalars(CreateProtoFieldData(*original_field));
            }
        } break;
        case DataType::FLOAT: {
            const auto* original_field = dynamic_cast<const FloatFieldData*>(&field);
            if (original_field) {
                field_data.set_allocated_scalars(CreateProtoFieldData(*original_field));
            }
        } break;
        case DataType::DOUBLE: {
            const auto* original_field = dynamic_cast<const DoubleFieldData*>(&field);
            if (original_field) {
                field_data.set_allocated_scalars(CreateProtoFieldData(*original_field));
            }
        } break;
        case DataType::STRING: {
            const auto* original_field = dynamic_cast<const StringFieldData*>(&field);
            if (original_field) {
                field_data.set_allocated_scalars(CreateProtoFieldData(*original_field));
            }
        } break;

        default:
            break;
    }

    return field_data;
}

FieldDataPtr
CreateMilvusFieldData(const milvus::proto::schema::FieldData& field_data) {
    auto field_type = field_data.type();
    const auto& name = field_data.field_name();

    switch (field_type) {
        case proto::schema::DataType::BinaryVector:
            return std::make_shared<BinaryVecFieldData>(
                name, BuildFieldDataVectors<uint8_t>(field_data.vectors().dim(), field_data.vectors().binary_vector()));

        case proto::schema::DataType::FloatVector:
            return std::make_shared<FloatVecFieldData>(
                name,
                BuildFieldDataVectors<float>(field_data.vectors().dim(), field_data.vectors().float_vector().data()));

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

        case proto::schema::DataType::String:
            return std::make_shared<StringFieldData>(
                name, BuildFieldDataScalars<std::string>(field_data.scalars().string_data().data()));
        default:
            break;
    }

    return nullptr;
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

}  // namespace milvus
