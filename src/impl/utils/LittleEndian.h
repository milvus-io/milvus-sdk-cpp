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

// Internal little-endian serialization helpers shared by the BloomFilter and RoaringBitmap
// envelope writers. Kept in one place so duplicate file-local definitions cannot collide
// when the two sources share a unity-build batch.
inline void
WriteU16LE(uint8_t* data, uint16_t value) {
    data[0] = static_cast<uint8_t>(value);
    data[1] = static_cast<uint8_t>(value >> 8);
}

inline void
WriteU32LE(uint8_t* data, uint32_t value) {
    data[0] = static_cast<uint8_t>(value);
    data[1] = static_cast<uint8_t>(value >> 8);
    data[2] = static_cast<uint8_t>(value >> 16);
    data[3] = static_cast<uint8_t>(value >> 24);
}

inline void
WriteU64LE(uint8_t* data, uint64_t value) {
    for (int i = 0; i < 8; i++) {
        data[i] = static_cast<uint8_t>(value >> (8 * i));
    }
}

inline uint32_t
ReadU32LE(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) | static_cast<uint32_t>(data[1]) << 8 | static_cast<uint32_t>(data[2]) << 16 |
           static_cast<uint32_t>(data[3]) << 24;
}

inline uint64_t
ReadU64LE(const uint8_t* data) {
    return static_cast<uint64_t>(data[0]) | static_cast<uint64_t>(data[1]) << 8 | static_cast<uint64_t>(data[2]) << 16 |
           static_cast<uint64_t>(data[3]) << 24 | static_cast<uint64_t>(data[4]) << 32 |
           static_cast<uint64_t>(data[5]) << 40 | static_cast<uint64_t>(data[6]) << 48 |
           static_cast<uint64_t>(data[7]) << 56;
}

}  // namespace milvus
