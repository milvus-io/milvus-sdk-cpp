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
BuildFieldDataVectors(size_t out_len, size_t in_len, const V* vectors_data, size_t offset, size_t count) {
    std::vector<std::vector<T>> data{};
    data.reserve(count);
    for (size_t i = offset; i < offset + count; i++) {
        std::vector<T> item{};
        item.resize(out_len);
        std::memcpy(item.data(), vectors_data + i * in_len, in_len * sizeof(V));
        data.emplace_back(std::move(item));
    }
    return data;
}

template <typename T, typename ScalarData>
std::vector<T>
BuildFieldDataScalars(const ScalarData& scalar_data, size_t offset, size_t count) {
    std::vector<T> data{};
    data.reserve(count);
    auto begin = scalar_data.begin();
    std::advance(begin, offset);
    auto end = begin;
    std::advance(end, count);
    std::copy(begin, end, std::back_inserter(data));
    return data;
}

template <typename T, typename ScalarData>
std::vector<T>
BuildFieldDataScalars(const ScalarData& scalar_data) {
    return BuildFieldDataScalars<T>(scalar_data, 0, scalar_data.size());
}

FieldDataPtr
BuildMilvusArrayFieldData(const std::string& name, const milvus::proto::schema::ArrayArray& array_field, int offset,
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
                arr.emplace_back(std::move(BuildFieldDataScalars<bool>((*begin).bool_data().data())));
            }
            return std::make_shared<ArrayBoolFieldData>(name, arr);
        }
        case proto::schema::DataType::Int8: {
            std::vector<ArrayInt8FieldData::ElementT> arr;
            for (; begin != end; begin++) {
                arr.emplace_back(std::move(BuildFieldDataScalars<int8_t>((*begin).int_data().data())));
            }
            return std::make_shared<ArrayInt8FieldData>(name, arr);
        }
        case proto::schema::DataType::Int16: {
            std::vector<ArrayInt16FieldData::ElementT> arr;
            for (; begin != end; begin++) {
                arr.emplace_back(std::move(BuildFieldDataScalars<int16_t>((*begin).int_data().data())));
            }
            return std::make_shared<ArrayInt16FieldData>(name, arr);
        }
        case proto::schema::DataType::Int32: {
            std::vector<ArrayInt32FieldData::ElementT> arr;
            for (; begin != end; begin++) {
                arr.emplace_back(std::move(BuildFieldDataScalars<int32_t>((*begin).int_data().data())));
            }
            return std::make_shared<ArrayInt32FieldData>(name, arr);
        }
        case proto::schema::DataType::Int64: {
            std::vector<ArrayInt64FieldData::ElementT> arr;
            for (; begin != end; begin++) {
                arr.emplace_back(std::move(BuildFieldDataScalars<int64_t>((*begin).long_data().data())));
            }
            return std::make_shared<ArrayInt64FieldData>(name, arr);
        }
        case proto::schema::DataType::Float: {
            std::vector<ArrayFloatFieldData::ElementT> arr;
            for (; begin != end; begin++) {
                arr.emplace_back(std::move(BuildFieldDataScalars<float>((*begin).float_data().data())));
            }
            return std::make_shared<ArrayFloatFieldData>(name, arr);
        }
        case proto::schema::DataType::Double: {
            std::vector<ArrayDoubleFieldData::ElementT> arr;
            for (; begin != end; begin++) {
                arr.emplace_back(std::move(BuildFieldDataScalars<double>((*begin).double_data().data())));
            }
            return std::make_shared<ArrayDoubleFieldData>(name, arr);
        }
        case proto::schema::DataType::VarChar: {
            std::vector<ArrayVarCharFieldData::ElementT> arr;
            for (; begin != end; begin++) {
                arr.emplace_back(std::move(BuildFieldDataScalars<std::string>((*begin).string_data().data())));
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
            size_t len = field_data.vectors().dim() / 8;
            std::vector<std::vector<uint8_t>> vectors = BuildFieldDataVectors<uint8_t, char>(
                len, len, field_data.vectors().binary_vector().data(), offset, count);
            return std::make_shared<BinaryVecFieldData>(name, std::move(vectors));
        }
        case proto::schema::DataType::FloatVector: {
            size_t len = field_data.vectors().dim();
            std::vector<FloatVecFieldData::ElementT> vectors = BuildFieldDataVectors<float, float>(
                len, len, field_data.vectors().float_vector().data().data(), offset, count);
            return std::make_shared<FloatVecFieldData>(name, std::move(vectors));
        }
        case proto::schema::DataType::Float16Vector: {
            size_t out_len = field_data.vectors().dim();
            size_t in_len = field_data.vectors().dim() * 2;
            std::vector<Float16VecFieldData::ElementT> vectors = BuildFieldDataVectors<uint16_t, char>(
                out_len, in_len, field_data.vectors().float16_vector().data(), offset, count);
            return std::make_shared<Float16VecFieldData>(name, std::move(vectors));
        }
        case proto::schema::DataType::BFloat16Vector: {
            size_t out_len = field_data.vectors().dim();
            size_t in_len = field_data.vectors().dim() * 2;
            std::vector<BFloat16VecFieldData::ElementT> vectors = BuildFieldDataVectors<uint16_t, char>(
                out_len, in_len, field_data.vectors().bfloat16_vector().data(), offset, count);
            return std::make_shared<BFloat16VecFieldData>(name, std::move(vectors));
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
                objects.emplace_back(std::move(nlohmann::json::parse(s)));
            }
            return std::make_shared<JSONFieldData>(name, BuildFieldDataScalars<nlohmann::json>(objects, offset, count));
        }

        case proto::schema::DataType::Array: {
            return BuildMilvusArrayFieldData(name, field_data.scalars().array_data(), offset, count);
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
        case proto::schema::DataType::BinaryVector: {
            size_t len = field_data.vectors().dim() / 8;
            size_t count = field_data.vectors().binary_vector().size() / len;
            std::vector<std::vector<uint8_t>> vectors =
                BuildFieldDataVectors<uint8_t, char>(len, len, field_data.vectors().binary_vector().data(), 0, count);
            return std::make_shared<BinaryVecFieldData>(name, std::move(vectors));
        }
        case proto::schema::DataType::FloatVector: {
            size_t len = field_data.vectors().dim();
            size_t count = field_data.vectors().float_vector().data().size() / len;
            std::vector<FloatVecFieldData::ElementT> vectors = BuildFieldDataVectors<float, float>(
                len, len, field_data.vectors().float_vector().data().data(), 0, count);
            return std::make_shared<FloatVecFieldData>(name, std::move(vectors));
        }
        case proto::schema::DataType::Float16Vector: {
            size_t out_len = field_data.vectors().dim();
            size_t in_len = field_data.vectors().dim() * 2;
            size_t count = field_data.vectors().float16_vector().size() / in_len;
            std::vector<Float16VecFieldData::ElementT> vectors = BuildFieldDataVectors<uint16_t, char>(
                out_len, in_len, field_data.vectors().float16_vector().data(), 0, count);
            return std::make_shared<Float16VecFieldData>(name, std::move(vectors));
        }
        case proto::schema::DataType::BFloat16Vector: {
            size_t out_len = field_data.vectors().dim();
            size_t in_len = field_data.vectors().dim() * 2;
            size_t count = field_data.vectors().bfloat16_vector().size() / in_len;
            std::vector<BFloat16VecFieldData::ElementT> vectors = BuildFieldDataVectors<uint16_t, char>(
                out_len, in_len, field_data.vectors().bfloat16_vector().data(), 0, count);
            return std::make_shared<BFloat16VecFieldData>(name, std::move(vectors));
        }

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
                objects.emplace_back(std::move(nlohmann::json::parse(s)));
            }
            return std::make_shared<JSONFieldData>(name, BuildFieldDataScalars<nlohmann::json>(objects));
        }

        case proto::schema::DataType::Array: {
            const auto& scalars_data = field_data.scalars().array_data();
            return BuildMilvusArrayFieldData(name, scalars_data, 0, scalars_data.data().size());
        }
        default:
            return nullptr;
    }
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

void
SetTargetVectors(const FieldDataPtr& target, milvus::proto::milvus::SearchRequest* rpc_request) {
    // placeholders
    proto::common::PlaceholderGroup placeholder_group;
    auto& placeholder_value = *placeholder_group.add_placeholders();
    placeholder_value.set_tag("$0");
    if (target->Type() == DataType::BINARY_VECTOR) {
        // bins
        placeholder_value.set_type(proto::common::PlaceholderType::BinaryVector);
        auto& vectors = dynamic_cast<BinaryVecFieldData&>(*target);
        for (const auto& vector : vectors.Data()) {
            std::string placeholder_data(reinterpret_cast<const char*>(vector.data()), vector.size());
            placeholder_value.add_values(std::move(placeholder_data));
        }
        rpc_request->set_nq(static_cast<int64_t>(vectors.Count()));
    } else if (target->Type() == DataType::FLOAT_VECTOR) {
        // floats
        placeholder_value.set_type(proto::common::PlaceholderType::FloatVector);
        auto& vectors = dynamic_cast<FloatVecFieldData&>(*target);
        for (const auto& vector : vectors.Data()) {
            std::string placeholder_data(reinterpret_cast<const char*>(vector.data()), vector.size() * sizeof(float));
            placeholder_value.add_values(std::move(placeholder_data));
        }
        rpc_request->set_nq(static_cast<int64_t>(vectors.Count()));
    } else if (target->Type() == DataType::SPARSE_FLOAT_VECTOR) {
        // sparse
        placeholder_value.set_type(proto::common::PlaceholderType::SparseFloatVector);
        auto& vectors = dynamic_cast<SparseFloatVecFieldData&>(*target);
        for (const auto& sparse : vectors.Data()) {
            std::string placeholder_data = EncodeSparseFloatVector(sparse);
            placeholder_value.add_values(std::move(placeholder_data));
        }
        rpc_request->set_nq(static_cast<int64_t>(vectors.Count()));
    } else if (target->Type() == DataType::FLOAT16_VECTOR) {
        // float16
        placeholder_value.set_type(proto::common::PlaceholderType::Float16Vector);
        auto& vectors = dynamic_cast<Float16VecFieldData&>(*target);
        for (const auto& vector : vectors.Data()) {
            std::string placeholder_data(reinterpret_cast<const char*>(vector.data()),
                                         vector.size() * sizeof(uint16_t));
            placeholder_value.add_values(std::move(placeholder_data));
        }
        rpc_request->set_nq(static_cast<int64_t>(vectors.Count()));
    } else if (target->Type() == DataType::BFLOAT16_VECTOR) {
        // float16
        placeholder_value.set_type(proto::common::PlaceholderType::BFloat16Vector);
        auto& vectors = dynamic_cast<BFloat16VecFieldData&>(*target);
        for (const auto& vector : vectors.Data()) {
            std::string placeholder_data(reinterpret_cast<const char*>(vector.data()),
                                         vector.size() * sizeof(uint16_t));
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
    }  // else throw an exception?
    rpc_request->set_placeholder_group(std::move(placeholder_group.SerializeAsString()));
}

void
SetExtraParams(const std::unordered_map<std::string, std::string>& params,
               milvus::proto::milvus::SearchRequest* rpc_request) {
    // offet/radius/range_filter/nprobe etc.
    // in old milvus versions, all extra params are under "params"
    // in new milvus versions, all extra params are in the top level
    nlohmann::json json_params;
    for (auto& pair : params) {
        auto kv_pair = rpc_request->add_search_params();
        kv_pair->set_key(pair.first);
        kv_pair->set_value(pair.second);

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
        auto kv_pair = rpc_request->add_search_params();
        kv_pair->set_key(PARAMS);
        kv_pair->set_value(json_params.dump());
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
GetRowsFromFieldsData(const std::vector<FieldDataPtr>& fields, EntityRows& rows) {
    rows.clear();
    size_t count = 0;
    auto status = GetRowCountOfFields(fields, count);
    if (!status.IsOk()) {
        return status;
    }

    auto getters = GenGetters(fields);
    for (auto i = 0; i < count; i++) {
        EntityRow row;
        for (auto& getter : getters) {
            row[getter.first] = getter.second(i);
        }
        rows.emplace_back(std::move(row));
    }
    return Status::OK();
}

Status
GetRowFromFieldsData(const std::vector<FieldDataPtr>& fields, size_t i, EntityRow& row) {
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

// current_db is the actual target db that the request is performed, for setting the GuaranteeTimestamp
// to compatible with old versions.
// for examples:
// - the MilvusClient connects to "my_db", the arguments.DatabaseName() is empty, target db is "my_db"
// - the MilvusClient connects to "", the arguments.DatabaseName() is empty, target db is "default"
// - the MilvusClient connects to "", the arguments.DatabaseName() is "my_db", target db is "my_db"
// - the MilvusClient connects to "db_1", the arguments.DatabaseName() is "db_2", target db is "db_2"
Status
ConvertQueryRequest(const QueryArguments& arguments, const std::string& current_db,
                    proto::milvus::QueryRequest& rpc_request) {
    auto db_name = arguments.DatabaseName();
    if (!db_name.empty()) {
        rpc_request.set_db_name(db_name);
    }
    rpc_request.set_collection_name(arguments.CollectionName());
    for (const auto& partition_name : arguments.PartitionNames()) {
        rpc_request.add_partition_names(partition_name);
    }

    rpc_request.set_expr(arguments.Filter());
    for (const auto& field : arguments.OutputFields()) {
        rpc_request.add_output_fields(field);
    }

    // limit/offet etc.
    auto& params = arguments.ExtraParams();
    for (auto& pair : params) {
        auto kv_pair = rpc_request.add_query_params();
        kv_pair->set_key(pair.first);
        kv_pair->set_value(pair.second);
    }

    ConsistencyLevel level = arguments.GetConsistencyLevel();
    uint64_t guarantee_ts = DeduceGuaranteeTimestamp(level, current_db, arguments.CollectionName());
    rpc_request.set_guarantee_timestamp(guarantee_ts);
    rpc_request.set_travel_timestamp(arguments.TravelTimestamp());

    if (level == ConsistencyLevel::NONE) {
        rpc_request.set_use_default_consistency(true);
    } else {
        rpc_request.set_consistency_level(ConsistencyLevelCast(arguments.GetConsistencyLevel()));
    }
    return Status::OK();
}

Status
ConvertQueryResults(const proto::milvus::QueryResults& rpc_results, QueryResults& results) {
    std::vector<milvus::FieldDataPtr> return_fields{};
    return_fields.reserve(rpc_results.fields_data_size());
    for (const auto& field_data : rpc_results.fields_data()) {
        return_fields.emplace_back(std::move(CreateMilvusFieldData(field_data)));
    }

    results = QueryResults(std::move(return_fields));
    return Status::OK();
}

// current_db is the actual target db that the request is performed, for setting the GuaranteeTimestamp
// to compatible with old versions.
// for examples:
// - the MilvusClient connects to "my_db", the arguments.DatabaseName() is empty, target db is "my_db"
// - the MilvusClient connects to "", the arguments.DatabaseName() is empty, target db is "default"
// - the MilvusClient connects to "", the arguments.DatabaseName() is "my_db", target db is "my_db"
// - the MilvusClient connects to "db_1", the arguments.DatabaseName() is "db_2", target db is "db_2"
Status
ConvertSearchRequest(const SearchArguments& arguments, const std::string& current_db,
                     proto::milvus::SearchRequest& rpc_request) {
    auto db_name = arguments.DatabaseName();
    if (!db_name.empty()) {
        rpc_request.set_db_name(db_name);
    }
    rpc_request.set_collection_name(arguments.CollectionName());
    rpc_request.set_dsl_type(proto::common::DslType::BoolExprV1);
    if (!arguments.Filter().empty()) {
        rpc_request.set_dsl(arguments.Filter());
    }
    for (const auto& partition_name : arguments.PartitionNames()) {
        rpc_request.add_partition_names(partition_name);
    }
    for (const auto& output_field : arguments.OutputFields()) {
        rpc_request.add_output_fields(output_field);
    }

    // set target vectors
    SetTargetVectors(arguments.TargetVectors(), &rpc_request);

    auto setParamFunc = [&rpc_request](const std::string& key, const std::string& value) {
        auto kv_pair = rpc_request.add_search_params();
        kv_pair->set_key(key);
        kv_pair->set_value(value);
    };

    // set anns field name, if the name is empty and the collection has only one vector field,
    // milvus server will use the vector field name as anns name. If the collection has multiple
    // vector fields, user needs to explicitly provide an anns field name.
    auto anns_field = arguments.AnnsField();
    if (!anns_field.empty()) {
        setParamFunc(milvus::ANNS_FIELD, anns_field);
    }

    // for history reason, query() requires "limit", search() requires "topk"
    setParamFunc(milvus::TOPK, std::to_string(arguments.Limit()));

    // set this value only when client specified, otherwise let server to get it from index parameters
    if (arguments.MetricType() != MetricType::DEFAULT) {
        setParamFunc(milvus::METRIC_TYPE, std::to_string(arguments.MetricType()));
    }

    // offset
    setParamFunc(milvus::OFFSET, std::to_string(arguments.Offset()));

    // round decimal
    setParamFunc(milvus::ROUND_DECIMAL, std::to_string(arguments.RoundDecimal()));

    // ignore growing
    setParamFunc(milvus::IGNORE_GROWING, arguments.IgnoreGrowing() ? "true" : "false");

    // group by
    auto group_by_field = arguments.GroupByField();
    if (!group_by_field.empty()) {
        setParamFunc(milvus::GROUPBY_FIELD, arguments.GroupByField());
    }

    // extra params radius/range_filter/nprobe etc.
    SetExtraParams(arguments.ExtraParams(), &rpc_request);

    // consistency level
    ConsistencyLevel level = arguments.GetConsistencyLevel();
    uint64_t guarantee_ts = DeduceGuaranteeTimestamp(level, current_db, arguments.CollectionName());
    rpc_request.set_guarantee_timestamp(guarantee_ts);
    rpc_request.set_travel_timestamp(arguments.TravelTimestamp());

    if (level == ConsistencyLevel::NONE) {
        rpc_request.set_use_default_consistency(true);
    } else {
        rpc_request.set_consistency_level(ConsistencyLevelCast(arguments.GetConsistencyLevel()));
    }
    return Status::OK();
}

// current_db is the actual target db that the request is performed, for setting the GuaranteeTimestamp
// to compatible with old versions.
// for examples:
// - the MilvusClient connects to "my_db", the arguments.DatabaseName() is empty, target db is "my_db"
// - the MilvusClient connects to "", the arguments.DatabaseName() is empty, target db is "default"
// - the MilvusClient connects to "", the arguments.DatabaseName() is "my_db", target db is "my_db"
// - the MilvusClient connects to "db_1", the arguments.DatabaseName() is "db_2", target db is "db_2"
Status
ConvertHybridSearchRequest(const HybridSearchArguments& arguments, const std::string& current_db,
                           proto::milvus::HybridSearchRequest& rpc_request) {
    auto db_name = arguments.DatabaseName();
    if (!db_name.empty()) {
        rpc_request.set_db_name(db_name);
    }
    rpc_request.set_collection_name(arguments.CollectionName());

    for (const auto& partition_name : arguments.PartitionNames()) {
        rpc_request.add_partition_names(partition_name);
    }
    for (const auto& output_field : arguments.OutputFields()) {
        rpc_request.add_output_fields(output_field);
    }

    for (const auto& sub_request : arguments.SubRequests()) {
        auto search_req = rpc_request.add_requests();
        SetTargetVectors(sub_request->TargetVectors(), search_req);

        // set filter expression
        search_req->set_dsl_type(proto::common::DslType::BoolExprV1);
        if (!sub_request->Filter().empty()) {
            search_req->set_dsl(sub_request->Filter());
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
        SetExtraParams(sub_request->ExtraParams(), search_req);
    }

    // set rerank/limit/offset/round decimal
    auto reranker = arguments.Rerank();
    auto params = reranker->Params();
    params[LIMIT] = std::to_string(arguments.Limit());  // hybrid search is new interface, requires "limit"
    params[OFFSET] = std::to_string(arguments.Offset());
    params[ROUND_DECIMAL] = std::to_string(arguments.RoundDecimal());
    params[IGNORE_GROWING] = arguments.IgnoreGrowing() ? "true" : "false";

    for (auto& pair : params) {
        auto kv_pair = rpc_request.add_rank_params();
        kv_pair->set_key(pair.first);
        kv_pair->set_value(pair.second);
    }

    // consistancy level
    ConsistencyLevel level = arguments.GetConsistencyLevel();
    uint64_t guarantee_ts = DeduceGuaranteeTimestamp(level, current_db, arguments.CollectionName());
    rpc_request.set_guarantee_timestamp(guarantee_ts);

    if (level == ConsistencyLevel::NONE) {
        rpc_request.set_use_default_consistency(true);
    } else {
        rpc_request.set_consistency_level(ConsistencyLevelCast(arguments.GetConsistencyLevel()));
    }
    return Status::OK();
}

Status
ConvertSearchResults(const proto::milvus::SearchResults& rpc_results, const std::string& pk_name,
                     SearchResults& results) {
    const auto& result_data = rpc_results.results();
    const auto& ids = result_data.ids();
    const auto& fields_data = result_data.fields_data();
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
            item_fields_data.emplace_back(std::move(milvus::CreateMilvusFieldData(field_data, offset, item_topk)));
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
            single_results.emplace_back(real_pk_name, score_name, std::move(item_fields_data));
        } catch (const std::exception& e) {
            return {StatusCode::UNKNOWN_ERROR, "Not able to parse search results, error: " + std::string(e.what())};
        }
        offset += item_topk;
    }

    results = SearchResults(std::move(single_results));
    return Status::OK();
}

template <typename T>
Status
CopyFieldDataRange(const FieldDataPtr& src, uint64_t from, uint64_t to, FieldDataPtr& target) {
    if (from >= to) {
        return {StatusCode::INVALID_AGUMENT, "illegal copy range"};
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
        return {StatusCode::INVALID_AGUMENT, "source field data is null pointer"};
    }
    if (from >= to || from >= src->Count()) {
        return {StatusCode::INVALID_AGUMENT, "invalid range to copy"};
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
        case DataType::VARCHAR: {
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
                case DataType::VARCHAR: {
                    return CopyFieldDataRange<ArrayVarCharFieldData>(src, from, to, target);
                }
                default:
                    return Status{StatusCode::INVALID_AGUMENT, "array field element type is unsupported"};
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
        default: {
            return Status{StatusCode::INVALID_AGUMENT, "field data type is unsupported"};
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
        return {StatusCode::INVALID_AGUMENT, "field data is null pointer"};
    }
    if ((from->Type() != to->Type()) || (from->ElementType() != to->ElementType())) {
        return {StatusCode::INVALID_AGUMENT, "not able to append data, type mismatch"};
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
        case DataType::VARCHAR: {
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
                case DataType::VARCHAR: {
                    return Append<ArrayVarCharFieldData>(from, to);
                }
                default:
                    return Status{StatusCode::INVALID_AGUMENT, "array field element type is unsupported"};
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
        default: {
            return Status{StatusCode::INVALID_AGUMENT, "field data type is unsupported"};
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

}  // namespace milvus
