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

#include "param/Field.h"

namespace milvus {
namespace param {
template <>
Field<bool>::Field(std::string name, std::vector<bool> values)
    : name_{std::move(name)}, type_{DataType::BOOL}, values_{std::move(values)} {
}

template <>
Field<int8_t>::Field(std::string name, std::vector<int8_t> values)
    : name_{std::move(name)}, type_{DataType::INT8}, values_{std::move(values)} {
}

template <>
Field<int16_t>::Field(std::string name, std::vector<int16_t> values)
    : name_{std::move(name)}, type_{DataType::INT16}, values_{std::move(values)} {
}

template <>
Field<int32_t>::Field(std::string name, std::vector<int32_t> values)
    : name_{std::move(name)}, type_{DataType::INT32}, values_{std::move(values)} {
}

template <>
Field<int64_t>::Field(std::string name, std::vector<int64_t> values)
    : name_{std::move(name)}, type_{DataType::INT64}, values_{std::move(values)} {
}

template <>
Field<float>::Field(std::string name, std::vector<float> values)
    : name_{std::move(name)}, type_{DataType::FLOAT}, values_{std::move(values)} {
}

template <>
Field<double>::Field(std::string name, std::vector<double> values)
    : name_{std::move(name)}, type_{DataType::DOUBLE}, values_{std::move(values)} {
}

template <>
Field<std::string>::Field(std::string name, std::vector<std::string> values)
    : name_{std::move(name)}, type_{DataType::STRING}, values_{std::move(values)} {
}

template <>
Field<std::vector<char>>::Field(std::string name, std::vector<std::vector<char>> values)
    : name_{std::move(name)}, type_{DataType::BINARY_VECTOR}, values_{std::move(values)} {
}

template <>
Field<std::vector<float>>::Field(std::string name, std::vector<std::vector<float>> values)
    : name_{std::move(name)}, type_{DataType::FLOAT_VECTOR}, values_{std::move(values)} {
}
}  // namespace param
}  // namespace milvus
