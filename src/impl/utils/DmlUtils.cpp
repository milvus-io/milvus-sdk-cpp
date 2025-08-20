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

#include "./GtsDict.h"
#include "./TypeUtils.h"
#include "milvus/types/Constants.h"
#include "milvus/utils/FP16.h"

namespace milvus {

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
            return Status{StatusCode::INVALID_AGUMENT, "Null pointer field is not allowed"};
        }

        auto it = std::find_if(collection_fields.begin(), collection_fields.end(),
                               [&field](const FieldSchema& schema) { return schema.Name() == field->Name(); });
        if (it != collection_fields.end()) {
            // the provided field is in collection schema, but it is not a required input
            // maybe the schema has been changed(primary key from auto-id to non-auto-id)
            // tell the MilvusClientImpl to update collection schema cache
            if (!IsInputField(*it, is_upsert)) {
                return Status{StatusCode::DATA_UNMATCH_SCHEMA, "No need to provide data for field: " + field->Name()};
            }

            // the provided field is not consistent with the schema
            if (field->Type() != it->FieldDataType()) {
                return Status{StatusCode::DATA_UNMATCH_SCHEMA, "Field data type mismatch for field: " + field->Name()};
            } else if (field->Type() == DataType::ARRAY && field->ElementType() != it->ElementType()) {
                return Status{StatusCode::DATA_UNMATCH_SCHEMA,
                              "Element data type mismatch for array field: " + field->Name()};
            }
            // accept it
            continue;
        }
        if (field->Name() == DYNAMIC_FIELD) {
            // if dynamic field is not JSON type, no need to update collection schema cache
            if (field->Type() != DataType::JSON) {
                return Status{StatusCode::INVALID_AGUMENT, "Require JSON data for dynamic field: " + field->Name()};
            }
            // if has dynamic field data but enable_dynamic_field is false, maybe the schema cache is out of date
            if (!enable_dynamic_field) {
                return Status{StatusCode::DATA_UNMATCH_SCHEMA, "Not a valid field: " + field->Name()};
            }
            // enable_dynamic_field is true and has dynamic field data
            // maybe the schema cache is out of date(enable_dynamic_field from true to false)
            // but we don't know, just pass the data to the server to check
            continue;
        }

        // redundant fields, maybe the schema has been changed(some fields added)
        // tell the MilvusClientImpl to update collection schema cache
        return Status{StatusCode::DATA_UNMATCH_SCHEMA, std::string(field->Name() + " is not a valid field")};
    }

    // this loop is for "are there any data missed?
    for (const auto& collection_field : collection_fields) {
        auto it = std::find_if(fields.begin(), fields.end(), [&collection_field](const FieldDataPtr& field) {
            return field->Name() == collection_field.Name();
        });

        if (it != fields.end()) {
            continue;
        }

        // some required fields are not provided, maybe the schema has been changed(some fields deleted)
        // tell the MilvusClientImpl to update collection schema cache
        if (IsInputField(collection_field, is_upsert)) {
            return Status{StatusCode::DATA_UNMATCH_SCHEMA, "Data is missed for field: " + collection_field.Name()};
        }
    }
    return Status::OK();
}

bool
IsRealFailure(const proto::common::Status& status) {
    // error_code() is legacy code, deprecated in v2.4, code() is new code returned by higher version milvus
    // both error_code() == RateLimit or code() == 8 means rate limit error
    return ((status.error_code() != proto::common::ErrorCode::RateLimit) &&
            (status.error_code() != proto::common::ErrorCode::Success)) ||
           (status.code() != 0 && status.code() != 8);
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

template <typename V, typename T>
Status
CheckValueRange(V val, const std::string& field_name) {
    T min = std::numeric_limits<T>::min();
    T max = std::numeric_limits<T>::max();
    if (val < static_cast<V>(min) || val > static_cast<V>(max)) {
        std::string err_msg = "Value " + std::to_string(val) + " should be in range [" + std::to_string(min) + ", " +
                              std::to_string(max) + "]";
        if (field_name.empty()) {
            err_msg += (" for field: " + field_name);
        }
        return Status{StatusCode::INVALID_AGUMENT, err_msg};
    }
    return Status::OK();
}

Status
CheckAndSetBinaryVector(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::VectorField* vf) {
    if (!obj.is_array()) {
        std::string err_msg = "Value type should be array for field: " + fs.Name();
        return Status{StatusCode::INVALID_AGUMENT, err_msg};
    }
    if (obj.size() * 8 != static_cast<std::size_t>(fs.Dimension())) {
        std::string err_msg = "Array length is not equal to dimension/8 for field: " + fs.Name();
        return Status{StatusCode::INVALID_AGUMENT, err_msg};
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
            return Status{StatusCode::INVALID_AGUMENT, err_msg};
        }
    }
    return Status::OK();
}

Status
CheckAndSetFloatVector(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::VectorField* vf) {
    if (!obj.is_array()) {
        std::string err_msg = "Value type should be array for field: " + fs.Name();
        return Status{StatusCode::INVALID_AGUMENT, err_msg};
    }
    if (obj.size() != static_cast<std::size_t>(fs.Dimension())) {
        std::string err_msg = "Array length is not equal to dimension for field: " + fs.Name();
        return Status{StatusCode::INVALID_AGUMENT, err_msg};
    }

    vf->set_dim(fs.Dimension());
    auto data = vf->mutable_float_vector()->mutable_data();
    for (const auto& ele : obj) {
        if (!ele.is_number_float()) {
            std::string err_msg = "Element value should be float for field: " + fs.Name();
            return Status{StatusCode::INVALID_AGUMENT, err_msg};
        }
        data->Add(ele.get<float>());
    }
    return Status::OK();
}

// We support two patterns of sparse vector:
// 1. a json dict like {"1": 0.1, "5": 0.2, "8": 0.15}
// 2. a json dict like {"indices": [1, 5, 8], "values": [0.1, 0.2, 0.15]}
Status
ParseSparseFloatVector(const nlohmann::json& obj, const std::string& field_name, std::map<uint32_t, float>& pairs) {
    if (!obj.is_object()) {
        std::cout << obj << std::endl;
        std::string err_msg = "Value type should be a dict for field: " + field_name;
        return Status{StatusCode::INVALID_AGUMENT, err_msg};
    }

    // parse indices/values from json
    std::vector<uint32_t> indices_vec;
    std::vector<float> values_vec;
    if (obj.contains(SPARSE_INDICES) && obj.contains(SPARSE_VALUES)) {
        const auto& indices = obj[SPARSE_INDICES];
        const auto& values = obj[SPARSE_VALUES];
        if (!indices.is_array() || !values.is_array()) {
            std::string err_msg = "Sparse indices or values must be array for field: " + field_name;
            return Status{StatusCode::INVALID_AGUMENT, err_msg};
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
                return Status{StatusCode::INVALID_AGUMENT, err_msg};
            }
        }
        for (const auto& val : values) {
            if (val.is_number()) {
                values_vec.push_back(val.get<float>());
            } else {
                std::string err_msg = "Values array should be numeric values for field: " + field_name;
                return Status{StatusCode::INVALID_AGUMENT, err_msg};
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
                return Status{StatusCode::INVALID_AGUMENT, err_msg};
            }

            auto val = pair.value();
            if (val.is_number()) {
                values_vec.push_back(val.get<float>());
            } else {
                std::string err_msg = "Values array should be numeric values for field: " + field_name;
                return Status{StatusCode::INVALID_AGUMENT, err_msg};
            }
        }
    }

    // avoid illegal input
    if (indices_vec.size() != values_vec.size()) {
        std::string err_msg = "Indices length(" + std::to_string(indices_vec.size()) +
                              ") is not equal to values length(" + std::to_string(values_vec.size()) +
                              ") for field: " + field_name;
        return Status{StatusCode::INVALID_AGUMENT, err_msg};
    }

    // the indices must be in accending order, and not allowed duplicated indices
    pairs.clear();
    for (auto i = 0; i < indices_vec.size(); i++) {
        pairs.insert(std::make_pair(indices_vec[i], values_vec[i]));
    }
    if (pairs.size() != indices_vec.size()) {
        return Status{StatusCode::INVALID_AGUMENT, "Duplicated indices for field: " + field_name};
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
        return Status{StatusCode::INVALID_AGUMENT, err_msg};
    }
    if (obj.size() != static_cast<std::size_t>(fs.Dimension())) {
        std::string err_msg = "Array length is not equal to dimension for field: " + fs.Name();
        return Status{StatusCode::INVALID_AGUMENT, err_msg};
    }

    bool is_bf16 = (fs.FieldDataType() == DataType::BFLOAT16_VECTOR);
    vf->set_dim(fs.Dimension());
    auto data = is_bf16 ? vf->mutable_bfloat16_vector() : vf->mutable_float16_vector();
    data->reserve(fs.Dimension() * 2);
    for (const auto& ele : obj) {
        if (!ele.is_number_float()) {
            std::string err_msg = "Element value should be float for field: " + fs.Name();
            return Status{StatusCode::INVALID_AGUMENT, err_msg};
        }

        float fval = ele.get<float>();
        // check value range
        // float16, the range is [-65504, +65504]
        // bfloat16, the range is almost equal to float32, no need to check
        if (!is_bf16 && (fval < -65504.0 || fval > 65504.0)) {
            std::string err_msg = "Value should be in range [-65504, 65504] for field: " + fs.Name();
            return Status{StatusCode::INVALID_AGUMENT, err_msg};
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
        return Status{StatusCode::INVALID_AGUMENT, "Value type should be array for field: " + fs.Name()};
    }
    if (obj.size() > static_cast<std::size_t>(fs.MaxCapacity())) {
        std::string error_msg =
            "Array length " + std::to_string(obj.size()) + " exceeds max capacity of field: " + fs.Name();
        return Status{StatusCode::INVALID_AGUMENT, error_msg};
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
CheckAndSetScalar(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::ScalarField* sf, bool is_array) {
    DataType dt = is_array ? fs.ElementType() : fs.FieldDataType();
    const std::string msg_prefix =
        is_array ? fs.Name() + " element type should be " : fs.Name() + " value type should be ";
    switch (dt) {
        case DataType::BOOL: {
            auto scalars = sf->mutable_bool_data()->mutable_data();
            if (!obj.is_boolean()) {
                return Status{StatusCode::INVALID_AGUMENT, msg_prefix + "bool"};
            }
            scalars->Add(obj.get<bool>());
            break;
        }
        case DataType::INT8:
        case DataType::INT16:
        case DataType::INT32: {
            if (!obj.is_number_integer() && !obj.is_number_unsigned()) {
                return Status{StatusCode::INVALID_AGUMENT, msg_prefix + "integer"};
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
                return Status{StatusCode::INVALID_AGUMENT, msg_prefix + "integer"};
            }
            auto scalars = sf->mutable_long_data()->mutable_data();
            scalars->Add(obj.get<int64_t>());
            break;
        }
        case DataType::FLOAT: {
            if (!obj.is_number()) {
                return Status{StatusCode::INVALID_AGUMENT, msg_prefix + "numeric"};
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
                return Status{StatusCode::INVALID_AGUMENT, msg_prefix + "numeric"};
            }
            auto scalars = sf->mutable_double_data()->mutable_data();
            scalars->Add(obj.get<double>());
            break;
        }
        case DataType::VARCHAR: {
            if (!obj.is_string()) {
                return Status{StatusCode::INVALID_AGUMENT, msg_prefix + "string"};
            }
            auto ss = obj.get<std::string>();
            if (ss.size() > fs.MaxLength()) {
                return Status{StatusCode::INVALID_AGUMENT, "Exceeds max length of field: " + fs.Name()};
            }
            auto scalars = sf->mutable_string_data()->mutable_data();
            scalars->Add(std::move(ss));
            break;
        }
        case DataType::JSON: {
            if (!obj.is_object() && !obj.is_array() && !obj.is_primitive()) {
                return Status{StatusCode::INVALID_AGUMENT, msg_prefix + "JSON"};
            }
            auto scalars = sf->mutable_json_data()->mutable_data();
            scalars->Add(obj.dump());
            break;
        }
        case DataType::ARRAY: {
            if (is_array) {
                return Status{StatusCode::INVALID_AGUMENT, "Not allow nested array for field: " + fs.Name()};
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
            return Status{StatusCode::INVALID_AGUMENT, err_msg};
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
            return CheckAndSetScalar(obj, fs, fd.mutable_scalars(), false);
    }
    return Status::OK();
}

Status
CheckAndSetRowData(const std::vector<nlohmann::json>& rows, const CollectionSchema& schema, bool is_upsert,
                   std::vector<proto::schema::FieldData>& rpc_fields) {
    const std::vector<FieldSchema>& schema_fields = schema.Fields();
    std::map<std::string, proto::schema::FieldData> name_fields;
    for (auto i = 0; i < rows.size(); i++) {
        const auto& row = rows[i];
        if (!row.is_object()) {
            return Status{StatusCode::INVALID_AGUMENT,
                          "The No." + std::to_string(i) + " input row is not a JSON dict object"};
        }

        for (const auto& field_schema : schema_fields) {
            const std::string& name = field_schema.Name();
            if (row.contains(name)) {
                // from v2.4.10, milvus allows upsert for auto-id pk, no need to check for upsert action
                if (field_schema.IsPrimaryKey() && field_schema.AutoID() && !is_upsert) {
                    return Status{StatusCode::INVALID_AGUMENT,
                                  "The primary key: " + name + " is auto generated, no need to input."};
                }
                proto::schema::FieldData& fd = name_fields[name];
                auto status = CheckAndSetFieldValue(row[name], field_schema, fd);
                if (!status.IsOk()) {
                    return status;
                }
            } else {
            }
        }
    }

    for (const auto& n : name_fields) {
        rpc_fields.emplace_back(n.second);
    }

    return Status::OK();
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
            return nlohmann::json(real_field->Value(i));
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
            case DataType::VARCHAR: {
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
                    case DataType::VARCHAR: {
                        getters.insert(std::make_pair(name, std::move(GenGetter<ArrayVarCharFieldData>(field))));
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
            return Status{StatusCode::INVALID_AGUMENT, "Row numbers of fields are not equal"};
        }
    }
    count = first_cnt;
    return Status::OK();
}

Status
GetRowsFromFieldsData(const std::vector<FieldDataPtr>& fields, std::vector<nlohmann::json>& rows) {
    rows.clear();
    size_t count = 0;
    auto status = GetRowCountOfFields(fields, count);
    if (!status.IsOk()) {
        return status;
    }

    auto getters = GenGetters(fields);
    for (auto i = 0; i < count; i++) {
        nlohmann::json row;
        for (auto& getter : getters) {
            row[getter.first] = getter.second(i);
        }
        rows.emplace_back(row);
    }
    return Status::OK();
}

Status
GetRowFromFieldsData(const std::vector<FieldDataPtr>& fields, size_t i, nlohmann::json& row) {
    row.clear();
    size_t count = 0;
    auto status = GetRowCountOfFields(fields, count);
    if (!status.IsOk()) {
        return status;
    }

    if (i >= count) {
        return Status{StatusCode::INVALID_AGUMENT, "out of bound"};
    }

    auto getters = GenGetters(fields);
    for (auto& getter : getters) {
        row[getter.first] = getter.second(i);
    }
    return Status::OK();
}

}  // namespace milvus
