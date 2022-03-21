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

#include "milvus/types/DistanceArray.h"

namespace milvus {

DistanceArray::DistanceArray() = default;

DistanceArray::DistanceArray(const std::vector<std::vector<int32_t>>& distance_array)
    : int_array_(distance_array), is_int_distance_{true} {
}

DistanceArray::DistanceArray(std::vector<std::vector<int32_t>>&& distance_array)
    : int_array_(std::move(distance_array)), is_int_distance_{true} {
}

DistanceArray::DistanceArray(const std::vector<std::vector<float>>& distance_array) : float_array_(distance_array) {
}

DistanceArray::DistanceArray(std::vector<std::vector<float>>&& distance_array)
    : float_array_(std::move(distance_array)) {
}

bool
DistanceArray::IsIntegerDistance() const {
    return is_int_distance_;
}

const std::vector<std::vector<int32_t>>&
DistanceArray::IntDistanceArray() const {
    return int_array_;
}

void
DistanceArray::SetIntDistance(const std::vector<std::vector<int32_t>>& distance_array) {
    int_array_ = distance_array;
    float_array_.clear();
    is_int_distance_ = true;
}

void
DistanceArray::SetIntDistance(std::vector<std::vector<int32_t>>&& distance_array) {
    int_array_ = std::move(distance_array);
    float_array_.clear();
    is_int_distance_ = true;
}

const std::vector<std::vector<float>>&
DistanceArray::FloatDistanceArray() const {
    return float_array_;
}

void
DistanceArray::SetFloatDistance(const std::vector<std::vector<float>>& distance_array) {
    float_array_ = distance_array;
    int_array_.clear();
    is_int_distance_ = false;
}

void
DistanceArray::SetFloatDistance(std::vector<std::vector<float>>&& distance_array) {
    float_array_ = std::move(distance_array);
    int_array_.clear();
    is_int_distance_ = false;
}

}  // namespace milvus
