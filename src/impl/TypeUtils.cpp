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
    if (!lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (!scalars.has_bool_data()) {
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
    if (!lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (!scalars.has_int_data()) {
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
    if (!lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (!scalars.has_int_data()) {
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
    if (!lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (!scalars.has_int_data()) {
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
    if (!lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (!scalars.has_long_data()) {
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
    if (!lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (!scalars.has_float_data()) {
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
    if (!lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (!scalars.has_double_data()) {
        return false;
    }
    const auto& scalars_data = scalars.double_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

bool
operator==(const proto::schema::FieldData& lhs, const VarCharFieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (!lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (!scalars.has_string_data()) {
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
    if (!lhs.has_vectors()) {
        return false;
    }

    const auto& vectors = lhs.vectors();
    if (vectors.has_float_vector()) {
        return false;
    }

    const auto& vectors_data = vectors.binary_vector();
    auto it = vectors_data.begin();
    const auto& strings = rhs.Data();
    for (const auto& s : strings) {
        for (const auto ch : s) {
            if (it == vectors_data.end() || *it != ch) {
                return false;
            }
            ++it;
        }
    }

    return it == vectors_data.end();
}

bool
operator==(const proto::schema::FieldData& lhs, const FloatVecFieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (!lhs.has_vectors()) {
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
        case DataType::VARCHAR:
            return lhs == dynamic_cast<const VarCharFieldData&>(rhs);
        case DataType::BINARY_VECTOR:
            return lhs == dynamic_cast<const BinaryVecFieldData&>(rhs);
        case DataType::FLOAT_VECTOR:
            return lhs == dynamic_cast<const FloatVecFieldData&>(rhs);
        default:
            return false;
    }
}

bool
operator==(const SegmentInfo& lhs, const SegmentInfo& rhs) {
    return lhs.CollectionID() == rhs.CollectionID() && lhs.PartitionID() == rhs.PartitionID() &&
           lhs.RowCount() == rhs.RowCount() && lhs.SegmentID() == rhs.SegmentID() && lhs.State() == rhs.State();
}

bool
operator==(const QuerySegmentInfo& lhs, const QuerySegmentInfo& rhs) {
    return lhs.CollectionID() == rhs.CollectionID() && lhs.PartitionID() == rhs.PartitionID() &&
           lhs.RowCount() == rhs.RowCount() && lhs.SegmentID() == rhs.SegmentID() && lhs.State() == rhs.State() &&
           lhs.IndexName() == rhs.IndexName() && lhs.IndexID() == rhs.IndexID() && lhs.NodeID() == rhs.NodeID();
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
        case DataType::VARCHAR:
            return proto::schema::DataType::VarChar;
        case DataType::BINARY_VECTOR:
            return proto::schema::DataType::BinaryVector;
        case DataType::FLOAT_VECTOR:
            return proto::schema::DataType::FloatVector;
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
        case proto::schema::DataType::BinaryVector:
            return DataType::BINARY_VECTOR;
        case proto::schema::DataType::FloatVector:
            return DataType::FLOAT_VECTOR;
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
    if (type == "HAMMING") {
        return MetricType::HAMMING;
    }
    if (type == "JACCARD") {
        return MetricType::JACCARD;
    }
    if (type == "TANIMOTO") {
        return MetricType::TANIMOTO;
    }
    if (type == "SUBSTRUCTURE") {
        return MetricType::SUBSTRUCTURE;
    }
    if (type == "SUPERSTRUCTURE") {
        return MetricType::SUPERSTRUCTURE;
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
    if (type == "IVF_HNSW") {
        return IndexType::IVF_HNSW;
    }
    if (type == "RHNSW_FLAT") {
        return IndexType::RHNSW_FLAT;
    }
    if (type == "RHNSW_SQ") {
        return IndexType::RHNSW_SQ;
    }
    if (type == "RHNSW_PQ") {
        return IndexType::RHNSW_PQ;
    }
    if (type == "ANNOY") {
        return IndexType::ANNOY;
    }
    if (type == "BIN_FLAT") {
        return IndexType::BIN_FLAT;
    }
    if (type == "BIN_IVF_FLAT") {
        return IndexType::BIN_IVF_FLAT;
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
        case DataType::BINARY_VECTOR:
            field_data.set_allocated_vectors(CreateProtoFieldData(dynamic_cast<const BinaryVecFieldData&>(field)));
            break;
        case DataType::FLOAT_VECTOR:
            field_data.set_allocated_vectors(CreateProtoFieldData(dynamic_cast<const FloatVecFieldData&>(field)));
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
        default:
            break;
    }

    return field_data;
}

FieldDataPtr
CreateMilvusFieldData(const milvus::proto::schema::FieldData& field_data, size_t offset, size_t count) {
    auto field_type = field_data.type();
    const auto& name = field_data.field_name();

    switch (field_type) {
        case proto::schema::DataType::BinaryVector:
            return std::make_shared<BinaryVecFieldData>(
                name, BuildFieldDataVectors<std::string>(field_data.vectors().dim() / 8,
                                                         field_data.vectors().binary_vector(), offset, count));

        case proto::schema::DataType::FloatVector:
            return std::make_shared<FloatVecFieldData>(
                name, BuildFieldDataVectors<std::vector<float>>(
                          field_data.vectors().dim(), field_data.vectors().float_vector().data(), offset, count));

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

}  // namespace milvus

namespace std {
std::string
to_string(milvus::MetricType metric_type) {
    switch (metric_type) {
        case milvus::MetricType::L2:
            return "L2";
        case milvus::MetricType::IP:
            return "IP";
        case milvus::MetricType::HAMMING:
            return "HAMMING";
        case milvus::MetricType::JACCARD:
            return "JACCARD";
        case milvus::MetricType::TANIMOTO:
            return "TANIMOTO";
        case milvus::MetricType::SUBSTRUCTURE:
            return "SUBSTRUCTURE";
        case milvus::MetricType::SUPERSTRUCTURE:
            return "SUPERSTRUCTURE";
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
        case milvus::IndexType::IVF_HNSW:
            return "IVF_HNSW";
        case milvus::IndexType::RHNSW_FLAT:
            return "RHNSW_FLAT";
        case milvus::IndexType::RHNSW_SQ:
            return "RHNSW_SQ";
        case milvus::IndexType::RHNSW_PQ:
            return "RHNSW_PQ";
        case milvus::IndexType::ANNOY:
            return "ANNOY";
        case milvus::IndexType::BIN_FLAT:
            return "BIN_FLAT";
        case milvus::IndexType::BIN_IVF_FLAT:
            return "BIN_IVF_FLAT";
        default:
            return "INVALID";
    }
}
}  // namespace std
