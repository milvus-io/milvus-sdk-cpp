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

#include <algorithm>
#include <iostream>
#include <map>
#include <random>
#include <type_traits>

#include "milvus/Status.h"
#include "milvus/types/Constants.h"
#include "milvus/utils/FP16.h"
#include "nlohmann/json.hpp"

namespace util {
using REAL_GEN = std::uniform_real_distribution<float>;
using INT_GEN = std::uniform_int_distribution<int>;

void
CheckStatus(std::string&& message, const milvus::Status& status) {
    if (status.IsOk()) {
        std::cout << "Succeed: " << message << std::endl;
    } else {
        std::cout << "Failed: " << message << " , status: " << status.Message() << std::endl;
        exit(1);
    }
}

std::vector<std::vector<float>>
GenerateFloatVectors(int dimension, int count) {
    std::random_device rd;
    std::mt19937 ran(rd());
    REAL_GEN float_gen(0.0, 1.0);
    std::vector<std::vector<float>> vectors(count);
    for (auto i = 0; i < count; ++i) {
        std::vector<float> vector(dimension);
        for (auto d = 0; d < dimension; ++d) {
            vector[d] = float_gen(ran);
        }
        vectors[i] = vector;
    }
    return std::move(vectors);
}

std::vector<float>
GenerateFloatVector(int dimension) {
    std::vector<std::vector<float>> vectors = GenerateFloatVectors(dimension, 1);
    return vectors[0];
}

std::vector<std::map<uint32_t, float>>
GenerateSparseVectors(int max_dim, int count) {
    std::random_device rd;
    std::mt19937 ran(rd());
    INT_GEN int_gen(0, max_dim);
    REAL_GEN float_gen(0.0, 1.0);
    std::vector<std::map<uint32_t, float>> vectors(count);
    for (auto i = 0; i < count; ++i) {
        int pair_count = int_gen(ran) % 10;
        std::map<uint32_t, float> vector{};
        for (auto d = 0; d < pair_count; ++d) {
            vector.insert(std::make_pair(int_gen(ran), float_gen(ran)));
        }
        vectors[i] = vector;
    }
    return std::move(vectors);
}

std::map<uint32_t, float>
GenerateSparseVector(int max_dim) {
    std::vector<std::map<uint32_t, float>> vectors = GenerateSparseVectors(max_dim, 1);
    return vectors[0];
}

nlohmann::json
GenerateSparseVectorInJson(int max_dim, bool key_value_pattern) {
    std::map<uint32_t, float> sparse = GenerateSparseVector(max_dim);

    // support two patterns of sparse
    // 1. a json dict like {"1": 0.1, "5": 0.2, "8": 0.15}
    // 2. a json dict like {"indices": [1, 5, 8], "values": [0.1, 0.2, 0.15]}
    nlohmann::json obj = nlohmann::json::object();
    if (key_value_pattern) {
        for (const auto& pair : sparse) {
            obj[std::to_string(pair.first)] = pair.second;
        }
    } else {
        std::vector<uint32_t> indices;
        std::vector<float> values;
        for (const auto& pair : sparse) {
            indices.push_back(pair.first);
            values.push_back(pair.second);
        }
        obj[milvus::SPARSE_INDICES] = indices;
        obj[milvus::SPARSE_VALUES] = values;
    }
    return obj;
}

std::vector<std::vector<uint8_t>>
GenerateBinaryVectors(int dimension, int count) {
    if (dimension % 8 != 0) {
        throw std::runtime_error("Binary vector dimension must be mod of 8!");
    }
    std::random_device rd;
    std::mt19937 ran(rd());
    INT_GEN int_gen(0, 256);
    int byte_count = dimension / 8;
    std::vector<std::vector<uint8_t>> vectors(count);
    for (auto i = 0; i < count; ++i) {
        std::vector<uint8_t> vector(byte_count);
        for (auto d = 0; d < byte_count; ++d) {
            vector[d] = int_gen(ran);
        }
        vectors[i] = vector;
    }
    return std::move(vectors);
}

std::vector<uint8_t>
GenerateBinaryVector(int dimension) {
    std::vector<std::vector<uint8_t>> vectors = GenerateBinaryVectors(dimension, 1);
    return vectors[0];
}

std::vector<uint16_t>
GenerateFloat16Vector(const std::vector<float>& src) {
    std::vector<uint16_t> vector(src.size());
    for (auto d = 0; d < src.size(); ++d) {
        vector[d] = milvus::F32toF16(src[d]);
    }
    return std::move(vector);
}

std::vector<std::vector<uint16_t>>
GenerateFloat16Vectors(int dimension, int count) {
    std::vector<std::vector<uint16_t>> vectors(count);
    for (auto i = 0; i < count; ++i) {
        auto src = GenerateFloatVector(dimension);
        vectors[i] = GenerateFloat16Vector(src);
    }
    return std::move(vectors);
}

std::vector<uint16_t>
GenerateFloat16Vector(int dimension) {
    std::vector<std::vector<uint16_t>> vectors = GenerateFloat16Vectors(dimension, 1);
    return vectors[0];
}

std::vector<uint16_t>
GenerateBFloat16Vector(const std::vector<float>& src) {
    std::vector<uint16_t> vector(src.size());
    for (auto d = 0; d < src.size(); ++d) {
        vector[d] = milvus::F32toBF16(src[d]);
    }
    return std::move(vector);
}

std::vector<std::vector<uint16_t>>
GenerateBFloat16Vectors(int dimension, int count) {
    std::vector<std::vector<uint16_t>> vectors(count);
    for (auto i = 0; i < count; ++i) {
        auto src = GenerateFloatVector(dimension);
        vectors[i] = GenerateBFloat16Vector(src);
    }
    return std::move(vectors);
}

std::vector<uint16_t>
GenerateBFloat16Vector(int dimension) {
    std::vector<std::vector<uint16_t>> vectors = GenerateBFloat16Vectors(dimension, 1);
    return vectors[0];
}

std::vector<std::vector<int8_t>>
GenerateInt8Vectors(int dimension, int count) {
    std::random_device rd;
    std::mt19937 ran(rd());
    INT_GEN int_gen(0, 256);
    std::vector<std::vector<int8_t>> vectors(count);
    for (auto i = 0; i < count; ++i) {
        std::vector<int8_t> vector(dimension);
        for (auto d = 0; d < dimension; ++d) {
            vector[d] = int_gen(ran);
        }
        vectors[i] = vector;
    }
    return std::move(vectors);
}

std::vector<int8_t>
GenerateInt8Vector(int dimension) {
    std::vector<std::vector<int8_t>> vectors = GenerateInt8Vectors(dimension, 1);
    return vectors[0];
}

template <typename T>
std::vector<T>
RandomeValues(T min, T max, int count) {
    std::random_device rd;
    std::mt19937 ran(rd());
    const auto is_float = std::is_floating_point<T>::value;
    typename std::conditional<is_float, REAL_GEN, INT_GEN>::type gen(min, max);

    std::vector<T> values(count);
    for (auto i = 0; i < count; ++i) {
        values[i] = static_cast<T>(gen(ran));
    }

    return std::move(values);
}

template <typename T>
T
RandomeValue(T min, T max) {
    std::vector<T> values = RandomeValues(min, max, 1);
    return values[0];
}

std::vector<bool>
RansomBools(int count) {
    auto values = RandomeValues<int>(0, 100, count);
    std::vector<bool> bools(count);
    std::transform(values.begin(), values.end(), bools.begin(), [](int x) { return x % 2 == 1; });
    return bools;
}

template <typename T>
void
PrintList(const std::vector<T>& obj) {
    std::cout << "[";
    auto it = obj.begin();
    while (it != obj.end()) {
        if (it != obj.begin()) {
            std::cout << ", ";
        }
        std::cout << *it;
        ++it;
    }
    std::cout << "]" << std::endl;
}

template <typename K, typename V>
void
PrintMap(const std::unordered_map<K, V>& obj) {
    std::cout << "{";
    auto it = obj.begin();
    while (it != obj.end()) {
        if (it != obj.begin()) {
            std::cout << ", ";
        }
        std::cout << it->first << ":" << it->second;
        ++it;
    }
    std::cout << "}" << std::endl;
}

}  // namespace util
