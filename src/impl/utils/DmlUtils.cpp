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

#include "DmlUtils.h"

#include <limits>
#include <set>

#include "./Constants.h"
#include "./GtsDict.h"
#include "./TypeUtils.h"
#include "milvus/types/Constants.h"
#include "milvus/utils/FP16.h"

namespace milvus {

template <typename V, typename T>
Status
CheckValueRange(V val, const std::string& field_name) {
    T min = std::numeric_limits<T>::lowest();
    T max = std::numeric_limits<T>::max();
    V v_max = static_cast<V>(max);
    V v_min = static_cast<V>(min);

    if (val < v_min || val > v_max) {
        std::string err_msg = "Value " + std::to_string(val) + " should be in range [" + std::to_string(v_min) + ", " +
                              std::to_string(v_max) + "]";
        if (!field_name.empty()) {
            err_msg += (" for field: " + field_name);
        }
        return {StatusCode::INVALID_AGUMENT, err_msg};
    }
    return Status::OK();
}

bool
IsInputField(const FieldSchema& field_schema, bool is_upsert) {
    // in v2.4, all the fields except the auto-id field are required for insert()
    // but in upsert(), all the fields including the auto-id field are requred to input
    if (field_schema.IsPrimaryKey() && field_schema.AutoID()) {
        return is_upsert;
    }
    // dynamic field is optional, not required by force
    if (field_schema.Name() == DYNAMIC_FIELD) {
        return false;
    }
    return true;
}

// The returned status error code affects the collection schema cache in MilvusClientImpl,
// carefully return the error code for different cases.
// DATA_UNMATCH_SCHEMA will tell the MilvusClientImpl to update collection schema cache,
// and call CheckInsertInput() to check the input again.
// Other error codes will be treated as failure immediatelly.
Status
CheckInsertInput(const CollectionDescPtr& collection_desc, const std::vector<FieldDataPtr>& fields, bool is_upsert) {
    bool enable_dynamic_field = collection_desc->Schema().EnableDynamicField();
    const auto& collection_fields = collection_desc->Schema().Fields();

    // this loop is for "are there any redundant data?"
    for (const auto& field : fields) {
        if (field == nullptr) {
            return {StatusCode::INVALID_AGUMENT, "Null pointer field is not allowed"};
        }

        auto it = std::find_if(collection_fields.begin(), collection_fields.end(),
                               [&field](const FieldSchema& schema) { return schema.Name() == field->Name(); });
        if (it != collection_fields.end()) {
            // the provided field is in collection schema, but it is not a required input
            // maybe the schema has been changed(primary key from auto-id to non-auto-id)
            // tell the MilvusClientImpl to update collection schema cache
            if (!IsInputField(*it, is_upsert)) {
                return {StatusCode::DATA_UNMATCH_SCHEMA, "No need to provide data for field: " + field->Name()};
            }

            // the provided field is not consistent with the schema
            if (field->Type() != it->FieldDataType()) {
                return {StatusCode::DATA_UNMATCH_SCHEMA, "Field data type mismatch for field: " + field->Name()};
            } else if (field->Type() == DataType::ARRAY && field->ElementType() != it->ElementType()) {
                return {StatusCode::DATA_UNMATCH_SCHEMA,
                        "Element data type mismatch for array field: " + field->Name()};
            }
            // accept it
            continue;
        }
        if (field->Name() == DYNAMIC_FIELD) {
            // if dynamic field is not JSON type, no need to update collection schema cache
            if (field->Type() != DataType::JSON) {
                return {StatusCode::INVALID_AGUMENT, "Require JSON data for dynamic field: " + field->Name()};
            }
            // if has dynamic field data but enable_dynamic_field is false, maybe the schema cache is out of date
            if (!enable_dynamic_field) {
                return {StatusCode::DATA_UNMATCH_SCHEMA, "Not a valid field: " + field->Name()};
            }
            // enable_dynamic_field is true and has dynamic field data
            // maybe the schema cache is out of date(enable_dynamic_field from true to false)
            // but we don't know, just pass the data to the server to check
            continue;
        }

        // redundant fields, maybe the schema has been changed(some fields added)
        // tell the MilvusClientImpl to update collection schema cache
        return {StatusCode::DATA_UNMATCH_SCHEMA, std::string(field->Name() + " is not a valid field")};
    }

    // this loop is for "are there any data missed?"
    for (const auto& collection_field : collection_fields) {
        auto it = std::find_if(fields.begin(), fields.end(), [&collection_field](const FieldDataPtr& field) {
            return field->Name() == collection_field.Name();
        });

        if (it != fields.end()) {
            continue;
        }

        // nullable field can be ignored
        if (collection_field.IsNullable()) {
            continue;
        }

        // the field has default value can be ignored
        if (!collection_field.DefaultValue().is_null()) {
            continue;
        }

        // some required fields are not provided, maybe the schema has been changed(some fields deleted)
        // tell the MilvusClientImpl to update collection schema cache
        if (IsInputField(collection_field, is_upsert)) {
            return {StatusCode::DATA_UNMATCH_SCHEMA, "Data is missed for field: " + collection_field.Name()};
        }
    }
    return Status::OK();
}

bool
IsRealFailure(const proto::common::Status& status) {
    // error_code() is legacy code, deprecated in v2.4, code() is new code returned by higher version milvus
    // both error_code() == RateLimit or code() == 8 means rate limit error
    auto legacy_code = status.error_code();  // compile warning at this line since proto deprecates this method
    return ((legacy_code != proto::common::ErrorCode::RateLimit) &&
            (legacy_code != proto::common::ErrorCode::Success)) ||
           (status.code() != 0 && status.code() != 8);
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

// We support two patterns of sparse vector:
// 1. a json dict like {"1": 0.1, "5": 0.2, "8": 0.15}
// 2. a json dict like {"indices": [1, 5, 8], "values": [0.1, 0.2, 0.15]}
Status
ParseSparseFloatVector(const nlohmann::json& obj, const std::string& field_name, std::map<uint32_t, float>& pairs) {
    if (!obj.is_object()) {
        std::cout << obj << std::endl;
        std::string err_msg = "Value type should be a dict for field: " + field_name;
        return {StatusCode::INVALID_AGUMENT, err_msg};
    }

    // parse indices/values from json
    std::vector<uint32_t> indices_vec;
    std::vector<float> values_vec;
    if (obj.contains(SPARSE_INDICES) && obj.contains(SPARSE_VALUES)) {
        const auto& indices = obj[SPARSE_INDICES];
        const auto& values = obj[SPARSE_VALUES];
        if (!indices.is_array() || !values.is_array()) {
            std::string err_msg = "Sparse indices or values must be array for field: " + field_name;
            return {StatusCode::INVALID_AGUMENT, err_msg};
        }
        for (const auto& index : indices) {
            if (index.is_number_integer() || index.is_number_unsigned()) {
                auto val = index.get<int64_t>();
                auto status = CheckValueRange<int64_t, uint32_t>(val, field_name);
                if (!status.IsOk()) {
                    return status;
                }
                indices_vec.push_back(static_cast<uint32_t>(val));
            } else {
                std::string err_msg = "Indices array should be integer values for field: " + field_name;
                return {StatusCode::INVALID_AGUMENT, err_msg};
            }
        }
        for (const auto& val : values) {
            if (val.is_number()) {
                values_vec.push_back(val.get<float>());
            } else {
                std::string err_msg = "Values array should be numeric values for field: " + field_name;
                return {StatusCode::INVALID_AGUMENT, err_msg};
            }
        }
    } else {
        for (const auto& pair : obj.items()) {
            try {
                auto index = static_cast<int64_t>(std::stoll(pair.key()));
                auto status = CheckValueRange<int64_t, uint32_t>(index, field_name);
                if (!status.IsOk()) {
                    return status;
                }
                indices_vec.push_back(static_cast<uint32_t>(index));
            } catch (...) {
                std::string err_msg = "Failed to parse index value'" + pair.key() + "' for field: " + field_name;
                return {StatusCode::INVALID_AGUMENT, err_msg};
            }

            auto val = pair.value();
            if (val.is_number()) {
                values_vec.push_back(val.get<float>());
            } else {
                std::string err_msg = "Values array should be numeric values for field: " + field_name;
                return {StatusCode::INVALID_AGUMENT, err_msg};
            }
        }
    }

    // avoid illegal input
    if (indices_vec.size() != values_vec.size()) {
        std::string err_msg = "Indices length(" + std::to_string(indices_vec.size()) +
                              ") is not equal to values length(" + std::to_string(values_vec.size()) +
                              ") for field: " + field_name;
        return {StatusCode::INVALID_AGUMENT, err_msg};
    }

    // the indices must be in accending order, and not allowed duplicated indices
    pairs.clear();
    for (auto i = 0; i < indices_vec.size(); i++) {
        pairs.insert(std::make_pair(indices_vec[i], values_vec[i]));
    }
    if (pairs.size() != indices_vec.size()) {
        return {StatusCode::INVALID_AGUMENT, "Duplicated indices for field: " + field_name};
    }

    return Status::OK();
}

////////////////////////////////////////////////////////////////////////////////////////
// methods to convert SDK field types to proto field types
proto::schema::VectorField*
CreateProtoVectorField(const BinaryVecFieldData& field) {
    auto ret = new proto::schema::VectorField{};
    auto& data = field.Data();
    auto dim = data.front().size() * 8;
    auto& vectors_data = *(ret->mutable_binary_vector());
    vectors_data.reserve(data.size() * dim);
    for (const auto& item : data) {
        std::copy(item.begin(), item.end(), std::back_inserter(vectors_data));
    }
    ret->set_dim(static_cast<int64_t>(dim));
    return ret;
}

proto::schema::VectorField*
CreateProtoVectorField(const FloatVecFieldData& field) {
    auto ret = new proto::schema::VectorField{};
    auto& data = field.Data();
    auto dim = data.front().size();
    auto& vectors_data = *(ret->mutable_float_vector()->mutable_data());
    vectors_data.Reserve(static_cast<int>(data.size() * dim));
    for (const auto& item : data) {
        vectors_data.Add(item.begin(), item.end());
    }
    ret->set_dim(static_cast<int64_t>(dim));
    return ret;
}

proto::schema::VectorField*
CreateProtoVectorField(const SparseFloatVecFieldData& field) {
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

proto::schema::VectorField*
CreateProtoVectorField(const Float16VecFieldData& field) {
    auto ret = new proto::schema::VectorField{};
    auto& data = field.Data();
    auto dim = data.front().size();
    auto vec_bytes = dim * 2;
    auto& vectors_data = *(ret->mutable_float16_vector());
    vectors_data.resize(data.size() * vec_bytes);
    for (size_t i = 0; i < data.size(); i++) {
        std::memcpy(&vectors_data[i * vec_bytes], data[i].data(), vec_bytes);
    }
    ret->set_dim(static_cast<int64_t>(dim));
    return ret;
}

proto::schema::VectorField*
CreateProtoVectorField(const BFloat16VecFieldData& field) {
    auto ret = new proto::schema::VectorField{};
    auto& data = field.Data();
    auto dim = data.front().size();
    auto vec_bytes = dim * 2;
    auto& vectors_data = *(ret->mutable_bfloat16_vector());
    vectors_data.resize(data.size() * vec_bytes);
    for (size_t i = 0; i < data.size(); i++) {
        std::memcpy(&vectors_data[i * vec_bytes], data[i].data(), vec_bytes);
    }
    ret->set_dim(static_cast<int64_t>(dim));
    return ret;
}

template <typename T>
void
CopyValidData(const Field& field, proto::schema::FieldData& proto_field) {
    auto actual_field = dynamic_cast<const T&>(field);
    const auto& valid_data = actual_field.ValidData();
    if (!valid_data.empty()) {
        auto ret = proto_field.mutable_valid_data();
        ret->Add(valid_data.begin(), valid_data.end());
    }
}

template <typename V, typename T>
T*
CreateProtoScalars(const Field& field, proto::schema::FieldData& proto_field, bool nullable) {
    auto actual_field = dynamic_cast<const V&>(field);
    const auto& data = actual_field.Data();
    auto ret = new T{};
    if (nullable) {
        CopyValidData<V>(field, proto_field);
        for (auto i = 0; i < actual_field.Count(); i++) {
            if (!actual_field.IsNull(i)) {
                ret->add_data(actual_field.Value(i));
            }
        }
    } else {
        ret->mutable_data()->Add(data.begin(), data.end());
    }
    return ret;
}

template <>
proto::schema::JSONArray*
CreateProtoScalars<JSONFieldData, proto::schema::JSONArray>(const Field& field, proto::schema::FieldData& proto_field,
                                                            bool nullable) {
    auto actual_field = dynamic_cast<const JSONFieldData&>(field);
    const auto& data = actual_field.Data();
    auto ret = new proto::schema::JSONArray{};
    if (nullable) {
        CopyValidData<JSONFieldData>(field, proto_field);
        for (auto i = 0; i < actual_field.Count(); i++) {
            if (!actual_field.IsNull(i)) {
                ret->mutable_data()->Add(actual_field.Value(i).dump());
            }
        }
    } else {
        for (const auto& item : data) {
            ret->add_data(item.dump());
        }
    }
    return ret;
}

Status
CreateProtoArrayField(const FieldDataSchema& data_schema, proto::schema::FieldData& proto_field) {
    const Field& field = *(data_schema.Data());
    const FieldSchemaPtr schema = data_schema.Schema();
    bool nullable_default = schema ? (schema->IsNullable() || !schema->DefaultValue().is_null()) : false;

    auto ret = proto_field.mutable_scalars();
    auto& array_data = *(ret->mutable_array_data());
    auto element_type = field.ElementType();
    array_data.set_element_type(DataTypeCast(element_type));

    switch (element_type) {
        case DataType::BOOL: {
            if (nullable_default) {
                CopyValidData<ArrayBoolFieldData>(field, proto_field);
            }
            const auto& actual_field = dynamic_cast<const ArrayBoolFieldData&>(field);
            for (auto i = 0; i < actual_field.Count(); i++) {
                if (!actual_field.IsNull(i)) {
                    auto row = actual_field.Value(i);
                    auto& scalar_field = *(array_data.add_data());
                    auto& scalars_data = *(scalar_field.mutable_bool_data()->mutable_data());
                    scalars_data.Add(row.begin(), row.end());
                }
            }
            break;
        }
        case DataType::INT8: {
            if (nullable_default) {
                CopyValidData<ArrayInt8FieldData>(field, proto_field);
            }
            const auto& actual_field = dynamic_cast<const ArrayInt8FieldData&>(field);
            for (auto i = 0; i < actual_field.Count(); i++) {
                if (!actual_field.IsNull(i)) {
                    auto row = actual_field.Value(i);
                    auto& scalar_field = *(array_data.add_data());
                    auto& scalars_data = *(scalar_field.mutable_int_data()->mutable_data());
                    scalars_data.Add(row.begin(), row.end());
                }
            }
            break;
        }
        case DataType::INT16: {
            if (nullable_default) {
                CopyValidData<ArrayInt16FieldData>(field, proto_field);
            }
            const auto& actual_field = dynamic_cast<const ArrayInt16FieldData&>(field);
            for (auto i = 0; i < actual_field.Count(); i++) {
                if (!actual_field.IsNull(i)) {
                    auto row = actual_field.Value(i);
                    auto& scalar_field = *(array_data.add_data());
                    auto& scalars_data = *(scalar_field.mutable_int_data()->mutable_data());
                    scalars_data.Add(row.begin(), row.end());
                }
            }
            break;
        }
        case DataType::INT32: {
            if (nullable_default) {
                CopyValidData<ArrayInt32FieldData>(field, proto_field);
            }
            const auto& actual_field = dynamic_cast<const ArrayInt32FieldData&>(field);
            for (auto i = 0; i < actual_field.Count(); i++) {
                if (!actual_field.IsNull(i)) {
                    auto row = actual_field.Value(i);
                    auto& scalar_field = *(array_data.add_data());
                    auto& scalars_data = *(scalar_field.mutable_int_data()->mutable_data());
                    scalars_data.Add(row.begin(), row.end());
                }
            }
            break;
        }
        case DataType::INT64: {
            if (nullable_default) {
                CopyValidData<ArrayInt64FieldData>(field, proto_field);
            }
            const auto& actual_field = dynamic_cast<const ArrayInt64FieldData&>(field);
            for (auto i = 0; i < actual_field.Count(); i++) {
                if (!actual_field.IsNull(i)) {
                    auto row = actual_field.Value(i);
                    auto& scalar_field = *(array_data.add_data());
                    auto& scalars_data = *(scalar_field.mutable_long_data()->mutable_data());
                    scalars_data.Add(row.begin(), row.end());
                }
            }
            break;
        }
        case DataType::FLOAT: {
            if (nullable_default) {
                CopyValidData<ArrayFloatFieldData>(field, proto_field);
            }
            const auto& actual_field = dynamic_cast<const ArrayFloatFieldData&>(field);
            for (auto i = 0; i < actual_field.Count(); i++) {
                if (!actual_field.IsNull(i)) {
                    auto row = actual_field.Value(i);
                    auto& scalar_field = *(array_data.add_data());
                    auto& scalars_data = *(scalar_field.mutable_float_data()->mutable_data());
                    scalars_data.Add(row.begin(), row.end());
                }
            }
            break;
        }
        case DataType::DOUBLE: {
            if (nullable_default) {
                CopyValidData<ArrayDoubleFieldData>(field, proto_field);
            }
            const auto& actual_field = dynamic_cast<const ArrayDoubleFieldData&>(field);
            for (auto i = 0; i < actual_field.Count(); i++) {
                if (!actual_field.IsNull(i)) {
                    auto row = actual_field.Value(i);
                    auto& scalar_field = *(array_data.add_data());
                    auto& scalars_data = *(scalar_field.mutable_double_data()->mutable_data());
                    scalars_data.Add(row.begin(), row.end());
                }
            }
            break;
        }
        case DataType::VARCHAR: {
            if (nullable_default) {
                CopyValidData<ArrayVarCharFieldData>(field, proto_field);
            }
            const auto& actual_field = dynamic_cast<const ArrayVarCharFieldData&>(field);
            for (auto i = 0; i < actual_field.Count(); i++) {
                if (!actual_field.IsNull(i)) {
                    auto row = actual_field.Value(i);
                    auto& scalar_field = *(array_data.add_data());
                    auto& scalars_data = *(scalar_field.mutable_string_data()->mutable_data());
                    scalars_data.Add(row.begin(), row.end());
                }
            }
            break;
        }
        default:
            return {StatusCode::NOT_SUPPORTED, "Unsupported array element type: " + std::to_string(element_type)};
    }
    return Status::OK();
}

Status
CreateProtoFieldData(const FieldDataSchema& data_schema, proto::schema::FieldData& field_data) {
    const Field& field = *(data_schema.Data());
    const FieldSchemaPtr schema = data_schema.Schema();
    bool nullable_default = schema ? (schema->IsNullable() || !schema->DefaultValue().is_null()) : false;

    auto scalar = field_data.mutable_scalars();
    const auto field_type = field.Type();
    field_data.set_field_name(field.Name());
    field_data.set_type(DataTypeCast(field_type));

    switch (field_type) {
        case DataType::BINARY_VECTOR:
            field_data.set_allocated_vectors(CreateProtoVectorField(dynamic_cast<const BinaryVecFieldData&>(field)));
            break;
        case DataType::FLOAT_VECTOR:
            field_data.set_allocated_vectors(CreateProtoVectorField(dynamic_cast<const FloatVecFieldData&>(field)));
            break;
        case DataType::SPARSE_FLOAT_VECTOR:
            field_data.set_allocated_vectors(
                CreateProtoVectorField(dynamic_cast<const SparseFloatVecFieldData&>(field)));
            break;
        case DataType::FLOAT16_VECTOR:
            field_data.set_allocated_vectors(CreateProtoVectorField(dynamic_cast<const Float16VecFieldData&>(field)));
            break;
        case DataType::BFLOAT16_VECTOR:
            field_data.set_allocated_vectors(CreateProtoVectorField(dynamic_cast<const BFloat16VecFieldData&>(field)));
            break;
        case DataType::BOOL:
            scalar->set_allocated_bool_data(
                CreateProtoScalars<BoolFieldData, proto::schema::BoolArray>(field, field_data, nullable_default));
            break;
        case DataType::INT8:
            scalar->set_allocated_int_data(
                CreateProtoScalars<Int8FieldData, proto::schema::IntArray>(field, field_data, nullable_default));
            break;
        case DataType::INT16:
            scalar->set_allocated_int_data(
                CreateProtoScalars<Int16FieldData, proto::schema::IntArray>(field, field_data, nullable_default));
            break;
        case DataType::INT32:
            scalar->set_allocated_int_data(
                CreateProtoScalars<Int32FieldData, proto::schema::IntArray>(field, field_data, nullable_default));
            break;
        case DataType::INT64:
            scalar->set_allocated_long_data(
                CreateProtoScalars<Int64FieldData, proto::schema::LongArray>(field, field_data, nullable_default));
            break;
        case DataType::FLOAT:
            scalar->set_allocated_float_data(
                CreateProtoScalars<FloatFieldData, proto::schema::FloatArray>(field, field_data, nullable_default));
            break;
        case DataType::DOUBLE:
            scalar->set_allocated_double_data(
                CreateProtoScalars<DoubleFieldData, proto::schema::DoubleArray>(field, field_data, nullable_default));
            break;
        case DataType::VARCHAR:
            scalar->set_allocated_string_data(
                CreateProtoScalars<VarCharFieldData, proto::schema::StringArray>(field, field_data, nullable_default));
            break;
        case DataType::JSON:
            scalar->set_allocated_json_data(
                CreateProtoScalars<JSONFieldData, proto::schema::JSONArray>(field, field_data, nullable_default));
            break;
        case DataType::ARRAY:
            return CreateProtoArrayField(data_schema, field_data);
        default:
            return {StatusCode::NOT_SUPPORTED, "Unsupported field type: " + std::to_string(field_type)};
    }

    return Status::OK();
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

Status
CheckAndSetBinaryVector(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::VectorField* vf) {
    if (!obj.is_array()) {
        std::string err_msg = "Value type should be array for field: " + fs.Name();
        return {StatusCode::INVALID_AGUMENT, err_msg};
    }
    if (obj.size() * 8 != static_cast<std::size_t>(fs.Dimension())) {
        std::string err_msg = "Array length is not equal to dimension/8 for field: " + fs.Name();
        return {StatusCode::INVALID_AGUMENT, err_msg};
    }

    vf->set_dim(fs.Dimension());
    auto data = vf->mutable_binary_vector();
    for (const auto& ele : obj) {
        if (ele.is_number_integer() || ele.is_number_unsigned()) {
            auto val = ele.get<int64_t>();
            auto status = CheckValueRange<int64_t, uint8_t>(val, fs.Name());
            if (!status.IsOk()) {
                return status;
            }
            data->push_back(static_cast<uint8_t>(val));
        } else {
            std::string err_msg = "Value should be int8 for field: " + fs.Name();
            return {StatusCode::INVALID_AGUMENT, err_msg};
        }
    }
    return Status::OK();
}

Status
CheckAndSetFloatVector(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::VectorField* vf) {
    if (!obj.is_array()) {
        std::string err_msg = "Value type should be array for field: " + fs.Name();
        return {StatusCode::INVALID_AGUMENT, err_msg};
    }
    if (obj.size() != static_cast<std::size_t>(fs.Dimension())) {
        std::string err_msg = "Array length is not equal to dimension for field: " + fs.Name();
        return {StatusCode::INVALID_AGUMENT, err_msg};
    }

    vf->set_dim(fs.Dimension());
    auto data = vf->mutable_float_vector()->mutable_data();
    for (const auto& ele : obj) {
        if (!ele.is_number_float()) {
            std::string err_msg = "Element value should be float for field: " + fs.Name();
            return {StatusCode::INVALID_AGUMENT, err_msg};
        }
        data->Add(ele.get<float>());
    }
    return Status::OK();
}

Status
CheckAndSetSparseFloatVector(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::VectorField* vf) {
    std::map<uint32_t, float> pairs;
    auto status = ParseSparseFloatVector(obj, fs.Name(), pairs);
    if (!status.IsOk()) {
        return status;
    }

    // convert indices/values to binary
    // indices are uint32_t type but the protobuf only has int32/int64, so we use int32 to store
    // the binary of uint32_t as both of them are 4 bits width.
    // value type is float 4 bits width, each pair of index/value is 8 bits, the binary length is N * 8 bits.
    auto contents = vf->mutable_sparse_float_vector()->add_contents();
    contents->reserve(pairs.size() * 8);
    for (auto& pair : pairs) {
        auto p1 = reinterpret_cast<const char*>(&pair.first);
        contents->append(p1, sizeof(uint32_t));

        auto p2 = reinterpret_cast<const char*>(&pair.second);
        contents->append(p2, sizeof(float));
    }

    return Status::OK();
}

Status
CheckAndSetFloat16Vector(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::VectorField* vf) {
    if (!obj.is_array()) {
        std::string err_msg = "Value type should be array for field: " + fs.Name();
        return {StatusCode::INVALID_AGUMENT, err_msg};
    }
    if (obj.size() != static_cast<std::size_t>(fs.Dimension())) {
        std::string err_msg = "Array length is not equal to dimension for field: " + fs.Name();
        return {StatusCode::INVALID_AGUMENT, err_msg};
    }

    bool is_bf16 = (fs.FieldDataType() == DataType::BFLOAT16_VECTOR);
    vf->set_dim(fs.Dimension());
    auto data = is_bf16 ? vf->mutable_bfloat16_vector() : vf->mutable_float16_vector();
    data->reserve(fs.Dimension() * 2);
    for (const auto& ele : obj) {
        if (!ele.is_number_float()) {
            std::string err_msg = "Element value should be float for field: " + fs.Name();
            return {StatusCode::INVALID_AGUMENT, err_msg};
        }

        float fval = ele.get<float>();
        // check value range
        // float16, the range is [-65504, +65504]
        // bfloat16, the range is almost equal to float32, no need to check
        if (!is_bf16 && (fval < -65504.0 || fval > 65504.0)) {
            std::string err_msg = "Value should be in range [-65504, 65504] for field: " + fs.Name();
            return {StatusCode::INVALID_AGUMENT, err_msg};
        }
        uint16_t val = is_bf16 ? F32toBF16(fval) : F32toF16(fval);
        auto p = reinterpret_cast<int8_t*>(&val);
        data->push_back(p[0]);
        data->push_back(p[1]);
    }
    return Status::OK();
}

Status
CheckAndSetArray(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::ArrayArray* aa) {
    if (!obj.is_array()) {
        return {StatusCode::INVALID_AGUMENT, "Value type should be array for field: " + fs.Name()};
    }
    if (obj.size() > static_cast<std::size_t>(fs.MaxCapacity())) {
        std::string error_msg =
            "Array length " + std::to_string(obj.size()) + " exceeds max capacity of field: " + fs.Name();
        return {StatusCode::INVALID_AGUMENT, error_msg};
    }
    if (aa->element_type() == proto::schema::DataType::None) {
        aa->set_element_type(DataTypeCast(fs.ElementType()));
    }
    auto scalars = aa->add_data();
    for (const auto& ele : obj) {
        auto status = CheckAndSetScalar(ele, fs, scalars, true);
        if (!status.IsOk()) {
            return status;
        }
    }
    return Status::OK();
}

Status
CheckAndSetNullableDefaultScalar(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::FieldData& fd) {
    // nullable and default value check
    // 1. if the field is nullable, user can input json_null/json_object(for row-based insert)
    //    1) if user input json_null, this value is replaced by default value
    //    2) if user input json_object, infer this value by type
    // 2. if the field is not nullable, user can input json_null/json_object(for row-based insert)
    //    1) if user input json_null, and default value is null, throw error
    //    2) if user input json_null, and default value is not null, this value is replaced by default value
    //    3) if user input json_object, infer this value by type
    nlohmann::json temp_obj = obj;
    if (fs.IsNullable()) {
        if (temp_obj.is_null()) {
            temp_obj = fs.DefaultValue();  // 1.1
        }
    } else {
        if (temp_obj.is_null()) {
            if (fs.DefaultValue().is_null()) {
                std::string msg = "Field " + fs.Name() + " is not nullable but the input value is null";
                return {StatusCode::INVALID_AGUMENT, msg};  // 2.1
            } else {
                temp_obj = fs.DefaultValue();  // 2.2
            }
        }
    }

    // assume we have a value list [1, 2, null, 3, null, 4]
    // the fd.valid_data is [true, true, false, true, false, true]
    // the fd.scalars is [1, 2, 3, 4]
    bool valid = !temp_obj.is_null();
    fd.mutable_valid_data()->Add(valid);

    // the fd.scalars only stores non-null values
    if (temp_obj.is_null()) {
        // for array field, we need to set the element type since CheckAndSetScalar() is not called
        if (fs.FieldDataType() == DataType::ARRAY) {
            fd.mutable_scalars()->mutable_array_data()->set_element_type(DataTypeCast(fs.ElementType()));
        }
        return Status::OK();
    }

    return CheckAndSetScalar(temp_obj, fs, fd.mutable_scalars(), false);
}

Status
CheckAndSetScalar(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::ScalarField* sf, bool is_array) {
    DataType dt = is_array ? fs.ElementType() : fs.FieldDataType();
    const std::string msg_prefix =
        is_array ? fs.Name() + " element type should be " : fs.Name() + " value type should be ";
    switch (dt) {
        case DataType::BOOL: {
            auto scalars = sf->mutable_bool_data()->mutable_data();
            if (!obj.is_boolean()) {
                return {StatusCode::INVALID_AGUMENT, msg_prefix + "bool"};
            }
            scalars->Add(obj.get<bool>());
            break;
        }
        case DataType::INT8:
        case DataType::INT16:
        case DataType::INT32: {
            if (!obj.is_number_integer() && !obj.is_number_unsigned()) {
                return {StatusCode::INVALID_AGUMENT, msg_prefix + "integer"};
            }
            auto val = obj.get<int64_t>();
            auto scalars = sf->mutable_int_data()->mutable_data();
            if (dt == DataType::INT8) {
                auto status = CheckValueRange<int64_t, int8_t>(val, fs.Name());
                if (!status.IsOk()) {
                    return status;
                }
                scalars->Add(static_cast<int8_t>(val));
            } else if (dt == DataType::INT16) {
                auto status = CheckValueRange<int64_t, int16_t>(val, fs.Name());
                if (!status.IsOk()) {
                    return status;
                }
                scalars->Add(static_cast<int16_t>(val));
            } else if (dt == DataType::INT32) {
                auto status = CheckValueRange<int64_t, int32_t>(val, fs.Name());
                if (!status.IsOk()) {
                    return status;
                }
                scalars->Add(static_cast<int32_t>(val));
            }
            break;
        }
        case DataType::INT64: {
            if (!obj.is_number_integer() && !obj.is_number_unsigned()) {
                return {StatusCode::INVALID_AGUMENT, msg_prefix + "integer"};
            }
            auto scalars = sf->mutable_long_data()->mutable_data();
            scalars->Add(obj.get<int64_t>());
            break;
        }
        case DataType::FLOAT: {
            if (!obj.is_number()) {
                return {StatusCode::INVALID_AGUMENT, msg_prefix + "numeric"};
            }
            auto val = obj.get<double>();
            auto scalars = sf->mutable_float_data()->mutable_data();
            auto status = CheckValueRange<double, float>(val, fs.Name());
            if (!status.IsOk()) {
                return status;
            }
            scalars->Add(obj.get<float>());
            break;
        }
        case DataType::DOUBLE: {
            if (!obj.is_number()) {
                return {StatusCode::INVALID_AGUMENT, msg_prefix + "numeric"};
            }
            auto scalars = sf->mutable_double_data()->mutable_data();
            scalars->Add(obj.get<double>());
            break;
        }
        case DataType::VARCHAR: {
            if (!obj.is_string()) {
                return {StatusCode::INVALID_AGUMENT, msg_prefix + "string"};
            }
            auto ss = obj.get<std::string>();
            if (ss.size() > fs.MaxLength()) {
                return {StatusCode::INVALID_AGUMENT, "Exceeds max length of field: " + fs.Name()};
            }
            auto scalars = sf->mutable_string_data()->mutable_data();
            scalars->Add(std::move(ss));
            break;
        }
        case DataType::JSON: {
            if (!obj.is_object() && !obj.is_array() && !obj.is_primitive()) {
                return {StatusCode::INVALID_AGUMENT, msg_prefix + "JSON"};
            }
            // for dynamic field, the json must be a dict
            // for the case user explicitly input $meta like {"id": 1, "vector": [], "$meta": {}}
            if (fs.Name() == DYNAMIC_FIELD && !obj.is_object()) {
                return {StatusCode::INVALID_AGUMENT, "'$meta' value must be a JSON dict"};
            }
            auto scalars = sf->mutable_json_data()->mutable_data();
            scalars->Add(obj.dump());
            break;
        }
        case DataType::ARRAY: {
            if (is_array) {
                return {StatusCode::INVALID_AGUMENT, "Not allow nested array for field: " + fs.Name()};
            }
            auto status = CheckAndSetArray(obj, fs, sf->mutable_array_data());
            if (!status.IsOk()) {
                return status;
            }
            break;
        }
        default: {
            std::string type_name = std::to_string(static_cast<int>(dt));
            std::string err_msg = is_array ? type_name + " is not supportted for field " + fs.Name()
                                           : type_name + " is not supportted in collection schema";
            return {StatusCode::INVALID_AGUMENT, err_msg};
        }
    }
    return Status::OK();
}

Status
CheckAndSetFieldValue(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::FieldData& fd) {
    DataType dt = fs.FieldDataType();
    fd.set_field_name(fs.Name());
    fd.set_type(DataTypeCast(dt));
    switch (dt) {
        case DataType::BINARY_VECTOR: {
            auto status = CheckAndSetBinaryVector(obj, fs, fd.mutable_vectors());
            if (!status.IsOk()) {
                return status;
            }
            break;
        }
        case DataType::FLOAT_VECTOR: {
            auto status = CheckAndSetFloatVector(obj, fs, fd.mutable_vectors());
            if (!status.IsOk()) {
                return status;
            }
            break;
        }
        case DataType::SPARSE_FLOAT_VECTOR: {
            auto status = CheckAndSetSparseFloatVector(obj, fs, fd.mutable_vectors());
            if (!status.IsOk()) {
                return status;
            }
            break;
        }
        case DataType::FLOAT16_VECTOR:
        case DataType::BFLOAT16_VECTOR: {
            auto status = CheckAndSetFloat16Vector(obj, fs, fd.mutable_vectors());
            if (!status.IsOk()) {
                return status;
            }
            break;
        }
        default:
            if (fs.IsNullable() || !fs.DefaultValue().is_null()) {
                return CheckAndSetNullableDefaultScalar(obj, fs, fd);
            } else {
                return CheckAndSetScalar(obj, fs, fd.mutable_scalars(), false);
            }
    }
    return Status::OK();
}

Status
CheckAndSetRowData(const EntityRows& rows, const CollectionSchema& schema, bool is_upsert,
                   std::vector<proto::schema::FieldData>& rpc_fields) {
    std::set<std::string> output_fields;
    for (const auto& function : schema.Functions()) {
        if (function) {
            for (const auto& name : function->OutputFieldNames()) {
                output_fields.insert(name);
            }
        }
    }

    const std::vector<FieldSchema>& schema_fields = schema.Fields();
    std::set<std::string> field_names;
    for (const auto& field_schema : schema_fields) {
        field_names.insert(field_schema.Name());
    }
    std::map<std::string, proto::schema::FieldData> proto_fields;

    // add a dynamic field into the output list if it is enabled
    if (schema.EnableDynamicField()) {
        proto::schema::FieldData dy;
        dy.set_type(milvus::proto::schema::DataType::JSON);
        dy.set_is_dynamic(true);
        proto_fields[DYNAMIC_FIELD] = dy;
    }

    for (auto i = 0; i < rows.size(); i++) {
        const auto& row = rows[i];
        if (!row.is_object()) {
            return {StatusCode::INVALID_AGUMENT,
                    "The No." + std::to_string(i) + " input row is not a JSON dict object"};
        }

        // process values for non-dynamic fields
        for (const auto& field_schema : schema_fields) {
            const std::string& name = field_schema.Name();
            nlohmann::json field_value;
            if (row.contains(name)) {
                field_value = row[name];
            } else {
                // if the field is auto-id, no need to provide value for insert, require to provide for upsert
                if ((field_schema.IsPrimaryKey() && field_schema.AutoID()) && !is_upsert) {
                    continue;
                }

                // if the field is an output field of doc-in-doc-out, no need to provide value
                if (output_fields.find(name) != output_fields.end()) {
                    continue;
                }

                // if the field doesn't have default value and is not nullable, require user to provide the value
                if (!field_schema.IsNullable() && field_schema.DefaultValue().is_null()) {
                    return {StatusCode::INVALID_AGUMENT, "The field: " + name + " is not provided."};
                }
            }

            // from v2.4.10, milvus allows upsert for auto-id pk, no need to check for upsert action
            if (field_schema.IsPrimaryKey() && field_schema.AutoID() && !is_upsert) {
                return {StatusCode::INVALID_AGUMENT,
                        "The primary key: " + name + " is auto generated, no need to input."};
            }

            proto::schema::FieldData& fd = proto_fields[name];
            auto status = CheckAndSetFieldValue(field_value, field_schema, fd);
            if (!status.IsOk()) {
                return status;
            }
        }

        // process values for dynamic fields
        if (schema.EnableDynamicField()) {
            nlohmann::json dynamic = nlohmann::json::object();
            for (auto it = row.begin(); it != row.end(); ++it) {
                if (field_names.find(it.key()) == field_names.end()) {
                    dynamic[it.key()] = it.value();
                }
            }
            auto sf = proto_fields[DYNAMIC_FIELD].mutable_scalars();
            auto scalars = sf->mutable_json_data()->mutable_data();
            scalars->Add(dynamic.dump());
        }
    }

    for (auto& n : proto_fields) {
        rpc_fields.emplace_back(std::move(n.second));
    }

    return Status::OK();
}

}  // namespace milvus
