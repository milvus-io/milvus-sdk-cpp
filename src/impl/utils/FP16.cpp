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

#include "milvus/utils/FP16.h"

#include <cmath>
#include <cstring>

namespace milvus {

uint16_t
F32toF16(float val) {
    uint32_t f32_bits = *reinterpret_cast<uint32_t*>(&val);
    uint16_t sign = (f32_bits >> 31) << 15;
    int32_t exponent = ((f32_bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = (f32_bits & 0x7FFFFF);

    if (std::isnan(val)) {
        return 0x7E00;  // NaN
    } else if (std::isinf(val)) {
        return sign | 0x7C00;  // Infinity
    } else if (val == 0.0f) {
        return 0;  // Zero
    }

    exponent += 15;
    if (exponent <= 0) {
        // return zero if denormal or underflow
        return sign;
    } else if (exponent >= 31) {
        // return Infinity if overflow
        return sign | 0x7C00;
    }

    return sign | ((exponent & 0x1F) << 10) | (mantissa >> 13);
}

float
F16toF32(uint16_t val) {
    unsigned int sign = (val & 0x8000) >> 15;
    unsigned int exponent = (val & 0x7C00) >> 10;
    unsigned int fraction = (val & 0x03FF);

    if (exponent == 0x1F) {
        if (fraction == 0) {
            return (sign == 0) ? INFINITY : -INFINITY;  // Infinity
        } else {
            return NAN;  // NaN
        }
    }

    auto f_fraction = static_cast<float>(fraction);
    if (exponent == 0) {
        if (fraction == 0) {
            return (sign == 0) ? 0.0f : -0.0f;  // Zero
        } else {
            // convert subnormal to normalized form for calculation
            float f32_val = f_fraction / 1024.0f * std::pow(2.0f, -14.0f);
            return (sign == 0) ? f32_val : -f32_val;
        }
    }

    auto f_exponent = static_cast<float>(exponent);
    float f32_val = (1.0f + f_fraction / 1024.0f) * std::pow(2.0f, f_exponent - 15.0f);
    return (sign == 0) ? f32_val : -f32_val;
}

uint16_t
F32toBF16(float val) {
    uint32_t f32_bits = *(reinterpret_cast<uint32_t*>(&val));

    // For bfloat16, we essentially take the upper 16 bits of the float32
    // This effectively truncates the lower 16 bits of the mantissa.
    // Note: This assumes a little-endian system where the higher-order bytes
    // of the float32 represent the exponent and most significant mantissa bits.
    // For big-endian, the byte order would need to be reversed or adjusted.
    return static_cast<uint16_t>(f32_bits >> 16);
}

float
BF16toF32(uint16_t val) {
    // Shift the 16-bit bfloat16 data left by 16 bits to align with float32's higher bits
    uint32_t f32_bits = static_cast<uint32_t>(val) << 16;

    // Reinterpret the resulting 32-bit integer as a float
    return *reinterpret_cast<float*>(&f32_bits);
}

}  // namespace milvus
