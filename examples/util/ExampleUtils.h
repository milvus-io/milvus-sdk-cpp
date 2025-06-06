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

#include <iostream>
#include <random>

#include "milvus/Status.h"

void
CheckStatus(std::string&& prefix, const milvus::Status& status) {
    if (!status.IsOk()) {
        std::cout << prefix << " " << status.Message() << std::endl;
        exit(1);
    }
}

std::vector<std::vector<float>>
GenerateFloatVectors(int dimension, int count) {
    std::random_device rd;
    std::mt19937 ran(rd());
    std::uniform_real_distribution<float> float_gen(0.0, 1.0);
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

template <typename T>
std::vector<T>
RandomeValues(T min, T max, int count) {
    std::random_device rd;
    std::mt19937 ran(rd());
    std::uniform_int_distribution<T> ran_gen(min, max);
    std::vector<T> values(count);
    for (auto i = 0; i < count; ++i) {
        values[i] = ran_gen(ran);
    }
    return std::move(values);
}

template <typename T>
T
RandomeValue(T min, T max) {
    std::vector<T> values = RandomeValues(min, max, 1);
    return values[0];
}
