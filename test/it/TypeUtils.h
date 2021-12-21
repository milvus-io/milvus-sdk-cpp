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

#include "milvus.pb.h"
#include "types/FieldData.h"

namespace milvus {

inline bool
operator==(const proto::schema::FieldData& lhs, const BoolFieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (not scalars.has_bool_data()) {
        return false;
    }
    const auto& scalars_data = scalars.bool_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

inline bool
operator==(const proto::schema::FieldData& lhs, const Int8FieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (not scalars.has_int_data()) {
        return false;
    }
    const auto& scalars_data = scalars.int_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

inline bool
operator==(const proto::schema::FieldData& lhs, const Int16FieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (not scalars.has_int_data()) {
        return false;
    }
    const auto& scalars_data = scalars.int_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

inline bool
operator==(const proto::schema::FieldData& lhs, const Int32FieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (not scalars.has_int_data()) {
        return false;
    }
    const auto& scalars_data = scalars.int_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

inline bool
operator==(const proto::schema::FieldData& lhs, const Int64FieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (not scalars.has_long_data()) {
        return false;
    }
    const auto& scalars_data = scalars.long_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

inline bool
operator==(const proto::schema::FieldData& lhs, const FloatFieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (not scalars.has_float_data()) {
        return false;
    }
    const auto& scalars_data = scalars.float_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

inline bool
operator==(const proto::schema::FieldData& lhs, const DoubleFieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (not scalars.has_double_data()) {
        return false;
    }
    const auto& scalars_data = scalars.double_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

inline bool
operator==(const proto::schema::FieldData& lhs, const StringFieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (not scalars.has_string_data()) {
        return false;
    }
    const auto& scalars_data = scalars.string_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

inline bool
operator==(const proto::schema::FieldData& lhs, const BinaryVecFieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_vectors()) {
        return false;
    }
    size_t dim = 0;
    if (rhs.Count() > 0) {
        dim = rhs.Data().front().size();
    }

    const auto& vectors = lhs.vectors();
    if (vectors.has_float_vector()) {
        return false;
    }
    const auto& vectors_data = vectors.binary_vector();
    if (vectors_data.size() != (rhs.Count() * dim)) {
        return false;
    }

    auto it = vectors_data.begin();
    for (const auto& item : rhs.Data()) {
        if (!std::equal(item.begin(), item.end(), it, [](uint8_t a, char b) { return static_cast<char>(a) == b; })) {
            return false;
        }
        std::advance(it, dim);
    }
    return true;
}

inline bool
operator==(const proto::schema::FieldData& lhs, const FloatVecFieldData& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (not lhs.has_vectors()) {
        return false;
    }
    size_t dim = 0;
    if (rhs.Count() > 0) {
        dim = rhs.Data().front().size();
    }

    const auto& vectors = lhs.vectors();
    if (!vectors.has_float_vector()) {
        return false;
    }
    const auto& vectors_data = vectors.float_vector().data();
    if (vectors_data.size() != (rhs.Count() * dim)) {
        return false;
    }

    auto it = vectors_data.begin();
    for (const auto& item : rhs.Data()) {
        if (!std::equal(item.begin(), item.end(), it)) {
            return false;
        }
        std::advance(it, dim);
    }
    return true;
}

}  // namespace milvus