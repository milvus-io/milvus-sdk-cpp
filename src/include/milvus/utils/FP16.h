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

#include <cstdint>

namespace milvus {

/**
 * @brief Convert a float32 value to a float16 value represented by uint16_t
 */
uint16_t
F32toF16(float val);

/**
 * @brief Convert a float16 value to a float32 value
 */
float
F16toF32(uint16_t val);

/**
 * @brief Convert a float32 value to a bfloat16 value represented by uint16_t
 */
uint16_t
F32toBF16(float val);

/**
 * @brief Convert a bfloat16 value to a float32 value
 */
float
BF16toF32(uint16_t val);

}  // namespace milvus
