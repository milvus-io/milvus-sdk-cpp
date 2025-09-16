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

#include "Utils.h"

#include <gtest/gtest.h>

namespace milvus {
const char* T_PK_NAME = "id";

void
BuildCollectionSchema(milvus::CollectionSchema& collection_schema) {
    collection_schema.AddField(milvus::FieldSchema(T_PK_NAME, milvus::DataType::INT64, "id", true, true));
    collection_schema.AddField(milvus::FieldSchema("bool", milvus::DataType::BOOL));
    collection_schema.AddField(milvus::FieldSchema("int8", milvus::DataType::INT8));
    collection_schema.AddField(milvus::FieldSchema("int16", milvus::DataType::INT16));
    collection_schema.AddField(milvus::FieldSchema("int32", milvus::DataType::INT32));
    collection_schema.AddField(milvus::FieldSchema("int64", milvus::DataType::INT64));
    collection_schema.AddField(milvus::FieldSchema("float", milvus::DataType::FLOAT));
    collection_schema.AddField(milvus::FieldSchema("double", milvus::DataType::DOUBLE));
    collection_schema.AddField(milvus::FieldSchema("varchar", milvus::DataType::VARCHAR));
    collection_schema.AddField(milvus::FieldSchema("json", milvus::DataType::JSON));
    collection_schema.AddField(
        milvus::FieldSchema("arr_bool", milvus::DataType::ARRAY).WithElementType(milvus::DataType::BOOL));
    collection_schema.AddField(
        milvus::FieldSchema("arr_int8", milvus::DataType::ARRAY).WithElementType(milvus::DataType::INT8));
    collection_schema.AddField(
        milvus::FieldSchema("arr_int16", milvus::DataType::ARRAY).WithElementType(milvus::DataType::INT16));
    collection_schema.AddField(
        milvus::FieldSchema("arr_int32", milvus::DataType::ARRAY).WithElementType(milvus::DataType::INT32));
    collection_schema.AddField(
        milvus::FieldSchema("arr_int64", milvus::DataType::ARRAY).WithElementType(milvus::DataType::INT64));
    collection_schema.AddField(
        milvus::FieldSchema("arr_float", milvus::DataType::ARRAY).WithElementType(milvus::DataType::FLOAT));
    collection_schema.AddField(
        milvus::FieldSchema("arr_double", milvus::DataType::ARRAY).WithElementType(milvus::DataType::DOUBLE));
    collection_schema.AddField(
        milvus::FieldSchema("arr_varchar", milvus::DataType::ARRAY).WithElementType(milvus::DataType::VARCHAR));
    collection_schema.AddField(
        milvus::FieldSchema("bin_vector", milvus::DataType::BINARY_VECTOR).WithDimension(T_DIMENSION));
    collection_schema.AddField(
        milvus::FieldSchema("float_vector", milvus::DataType::FLOAT_VECTOR).WithDimension(T_DIMENSION));
    collection_schema.AddField(
        milvus::FieldSchema("f16_vector", milvus::DataType::FLOAT16_VECTOR).WithDimension(T_DIMENSION));
    collection_schema.AddField(
        milvus::FieldSchema("bf16_vector", milvus::DataType::BFLOAT16_VECTOR).WithDimension(T_DIMENSION));
    collection_schema.AddField(milvus::FieldSchema("sparse_vector", milvus::DataType::SPARSE_FLOAT_VECTOR));
}

void
BuildFieldsData(const milvus::CollectionSchema& schema, std::vector<milvus::FieldDataPtr>& fields_data, int row_count) {
    fields_data.clear();
    for (const auto& field : schema.Fields()) {
        const auto& field_name = field.Name();
        switch (field.FieldDataType()) {
            case milvus::DataType::BOOL: {
                std::vector<bool> data;
                data.reserve(row_count);
                for (int i = 0; i < row_count; i++) {
                    data.push_back(i % 2 == 0);
                }
                auto ptr = std::make_shared<milvus::BoolFieldData>(field_name, std::move(data));
                fields_data.emplace_back(std::move(ptr));
                break;
            }
            case milvus::DataType::INT8: {
                std::vector<int8_t> data;
                data.reserve(row_count);
                for (int i = 0; i < row_count; i++) {
                    data.push_back(i % 256);
                }
                auto ptr = std::make_shared<milvus::Int8FieldData>(field_name, std::move(data));
                fields_data.emplace_back(std::move(ptr));
                break;
            }
            case milvus::DataType::INT16: {
                std::vector<int16_t> data;
                data.reserve(row_count);
                for (int i = 0; i < row_count; i++) {
                    data.push_back(i % 32768);
                }
                auto ptr = std::make_shared<milvus::Int16FieldData>(field_name, std::move(data));
                fields_data.emplace_back(std::move(ptr));
                break;
            }
            case milvus::DataType::INT32: {
                std::vector<int32_t> data;
                data.reserve(row_count);
                for (int i = 0; i < row_count; i++) {
                    data.push_back(i % 100000);
                }
                auto ptr = std::make_shared<milvus::Int32FieldData>(field_name, std::move(data));
                fields_data.emplace_back(std::move(ptr));
                break;
            }
            case milvus::DataType::INT64: {
                std::vector<int64_t> data;
                data.reserve(row_count);
                for (int i = 0; i < row_count; i++) {
                    data.push_back(i % 2000000);
                }
                auto ptr = std::make_shared<milvus::Int64FieldData>(field_name, std::move(data));
                fields_data.emplace_back(std::move(ptr));
                break;
            }
            case milvus::DataType::FLOAT: {
                std::vector<float> data;
                data.reserve(row_count);
                for (int i = 0; i < row_count; i++) {
                    data.push_back(i / 4);
                }
                auto ptr = std::make_shared<milvus::FloatFieldData>(field_name, std::move(data));
                fields_data.emplace_back(std::move(ptr));
                break;
            }
            case milvus::DataType::DOUBLE: {
                std::vector<double> data;
                data.reserve(row_count);
                for (int i = 0; i < row_count; i++) {
                    data.push_back(i / 3);
                }
                auto ptr = std::make_shared<milvus::DoubleFieldData>(field_name, std::move(data));
                fields_data.emplace_back(std::move(ptr));
                break;
            }
            case milvus::DataType::VARCHAR: {
                std::vector<std::string> data;
                data.reserve(row_count);
                for (int i = 0; i < row_count; i++) {
                    data.push_back("varchar_" + std::to_string(i));
                }
                auto ptr = std::make_shared<milvus::VarCharFieldData>(field_name, std::move(data));
                fields_data.emplace_back(std::move(ptr));
                break;
            }
            case milvus::DataType::JSON: {
                std::vector<nlohmann::json> data;
                data.reserve(row_count);
                for (int i = 0; i < row_count; i++) {
                    data.push_back({{"k", i}});
                }
                auto ptr = std::make_shared<milvus::JSONFieldData>(field_name, std::move(data));
                fields_data.emplace_back(std::move(ptr));
                break;
            }
            case milvus::DataType::ARRAY:
                switch (field.ElementType()) {
                    case milvus::DataType::BOOL: {
                        std::vector<std::vector<bool>> data;
                        data.reserve(row_count);
                        for (int i = 0; i < row_count; i++) {
                            data.push_back({i % 3 == 0, i % 4 == 0});
                        }
                        auto ptr = std::make_shared<milvus::ArrayBoolFieldData>(field_name, std::move(data));
                        fields_data.emplace_back(std::move(ptr));
                        break;
                    }
                    case milvus::DataType::INT8: {
                        std::vector<std::vector<int8_t>> data;
                        data.reserve(row_count);
                        for (int i = 0; i < row_count; i++) {
                            data.push_back({static_cast<int8_t>(i % 256), static_cast<int8_t>(i % 25)});
                        }
                        auto ptr = std::make_shared<milvus::ArrayInt8FieldData>(field_name, std::move(data));
                        fields_data.emplace_back(std::move(ptr));
                        break;
                    }
                    case milvus::DataType::INT16: {
                        std::vector<std::vector<int16_t>> data;
                        data.reserve(row_count);
                        for (int i = 0; i < row_count; i++) {
                            data.push_back({static_cast<int16_t>(i % 32768), static_cast<int16_t>(i % 32768)});
                        }
                        auto ptr = std::make_shared<milvus::ArrayInt16FieldData>(field_name, std::move(data));
                        fields_data.emplace_back(std::move(ptr));
                        break;
                    }
                    case milvus::DataType::INT32: {
                        std::vector<std::vector<int32_t>> data;
                        data.reserve(row_count);
                        for (int i = 0; i < row_count; i++) {
                            data.push_back({static_cast<int32_t>(i % 50000), static_cast<int32_t>(i % 50000)});
                        }
                        auto ptr = std::make_shared<milvus::ArrayInt32FieldData>(field_name, std::move(data));
                        fields_data.emplace_back(std::move(ptr));
                        break;
                    }
                    case milvus::DataType::INT64: {
                        std::vector<std::vector<int64_t>> data;
                        data.reserve(row_count);
                        for (int i = 0; i < row_count; i++) {
                            data.push_back({static_cast<int64_t>(i % 100000), static_cast<int64_t>(i % 100000)});
                        }
                        auto ptr = std::make_shared<milvus::ArrayInt64FieldData>(field_name, std::move(data));
                        fields_data.emplace_back(std::move(ptr));
                        break;
                    }
                    case milvus::DataType::FLOAT: {
                        std::vector<std::vector<float>> data;
                        data.reserve(row_count);
                        for (int i = 0; i < row_count; i++) {
                            data.push_back({static_cast<float>(i / 2), static_cast<float>(i / 5)});
                        }
                        auto ptr = std::make_shared<milvus::ArrayFloatFieldData>(field_name, std::move(data));
                        fields_data.emplace_back(std::move(ptr));
                        break;
                    }
                    case milvus::DataType::DOUBLE: {
                        std::vector<std::vector<double>> data;
                        data.reserve(row_count);
                        for (int i = 0; i < row_count; i++) {
                            data.push_back({static_cast<double>(i / 3), static_cast<double>(i / 4)});
                        }
                        auto ptr = std::make_shared<milvus::ArrayDoubleFieldData>(field_name, std::move(data));
                        fields_data.emplace_back(std::move(ptr));
                        break;
                    }
                    case milvus::DataType::VARCHAR: {
                        std::vector<std::vector<std::string>> data;
                        data.reserve(row_count);
                        for (int i = 0; i < row_count; i++) {
                            data.push_back({std::to_string(i % 128), std::to_string(i % 25)});
                        }
                        auto ptr = std::make_shared<milvus::ArrayVarCharFieldData>(field_name, std::move(data));
                        fields_data.emplace_back(std::move(ptr));
                        break;
                    }
                    default:
                        EXPECT_TRUE(false);
                        return;
                }
                break;
            case milvus::DataType::BINARY_VECTOR: {
                std::vector<std::vector<uint8_t>> data;
                data.reserve(row_count);
                for (int i = 0; i < row_count; i++) {
                    data.push_back({static_cast<uint8_t>(i % 128), static_cast<uint8_t>(i % 25)});
                }
                auto ptr = std::make_shared<milvus::BinaryVecFieldData>(field_name, std::move(data));
                fields_data.emplace_back(std::move(ptr));
                break;
            }
            case milvus::DataType::FLOAT_VECTOR: {
                std::vector<std::vector<float>> data;
                data.reserve(row_count);
                for (int i = 0; i < row_count; i++) {
                    data.push_back({static_cast<float>(i / 2), static_cast<float>(i / 5)});
                }
                auto ptr = std::make_shared<milvus::FloatVecFieldData>(field_name, std::move(data));
                fields_data.emplace_back(std::move(ptr));
                break;
            }
            case milvus::DataType::SPARSE_FLOAT_VECTOR: {
                std::vector<std::map<uint32_t, float>> data;
                data.reserve(row_count);
                for (int i = 0; i < row_count; i++) {
                    std::map<uint32_t, float> sparse;
                    sparse[i] = i / 3;
                    sparse[i * 2] = i / 5;
                    data.emplace_back(std::move(sparse));
                }
                auto ptr = std::make_shared<milvus::SparseFloatVecFieldData>(field_name, std::move(data));
                fields_data.emplace_back(std::move(ptr));
                break;
            }
            case milvus::DataType::FLOAT16_VECTOR: {
                std::vector<std::vector<uint16_t>> data;
                data.reserve(row_count);
                for (int i = 0; i < row_count; i++) {
                    data.push_back({static_cast<uint16_t>(i % 1000), static_cast<uint16_t>(i % 2000)});
                }
                auto ptr = std::make_shared<milvus::Float16VecFieldData>(field_name, std::move(data));
                fields_data.emplace_back(std::move(ptr));
                break;
            }
            case milvus::DataType::BFLOAT16_VECTOR: {
                std::vector<std::vector<uint16_t>> data;
                data.reserve(row_count);
                for (int i = 0; i < row_count; i++) {
                    data.push_back({static_cast<uint16_t>(i % 2000), static_cast<uint16_t>(i % 1000)});
                }
                auto ptr = std::make_shared<milvus::BFloat16VecFieldData>(field_name, std::move(data));
                fields_data.emplace_back(std::move(ptr));
                break;
            }
            default:
                EXPECT_TRUE(false);
                return;
        }
    }
}

}  // namespace milvus
