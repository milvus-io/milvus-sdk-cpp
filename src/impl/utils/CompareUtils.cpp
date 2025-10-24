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

#include "CompareUtils.h"

#include "./TypeUtils.h"

namespace milvus {

bool
IsEqual(const proto::schema::FieldData& lhs, const Field& rhs) {
    if (lhs.field_name() != rhs.Name()) {
        return false;
    }
    if (lhs.type() != DataTypeCast(rhs.Type())) {
        return false;
    }
    return true;
}

bool
operator==(const proto::schema::FieldData& lhs, const BoolFieldData& rhs) {
    if (!IsEqual(lhs, reinterpret_cast<const Field&>(rhs))) {
        return false;
    }
    if (!lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (!scalars.has_bool_data()) {
        return false;
    }
    const auto& scalars_data = scalars.bool_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

bool
operator==(const proto::schema::FieldData& lhs, const Int8FieldData& rhs) {
    if (!IsEqual(lhs, reinterpret_cast<const Field&>(rhs))) {
        return false;
    }
    if (!lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (!scalars.has_int_data()) {
        return false;
    }
    const auto& scalars_data = scalars.int_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

bool
operator==(const proto::schema::FieldData& lhs, const Int16FieldData& rhs) {
    if (!IsEqual(lhs, reinterpret_cast<const Field&>(rhs))) {
        return false;
    }
    if (!lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (!scalars.has_int_data()) {
        return false;
    }
    const auto& scalars_data = scalars.int_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

bool
operator==(const proto::schema::FieldData& lhs, const Int32FieldData& rhs) {
    if (!IsEqual(lhs, reinterpret_cast<const Field&>(rhs))) {
        return false;
    }
    if (!lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (!scalars.has_int_data()) {
        return false;
    }
    const auto& scalars_data = scalars.int_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

bool
operator==(const proto::schema::FieldData& lhs, const Int64FieldData& rhs) {
    if (!IsEqual(lhs, reinterpret_cast<const Field&>(rhs))) {
        return false;
    }
    if (!lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (!scalars.has_long_data()) {
        return false;
    }
    const auto& scalars_data = scalars.long_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

bool
operator==(const proto::schema::FieldData& lhs, const FloatFieldData& rhs) {
    if (!IsEqual(lhs, reinterpret_cast<const Field&>(rhs))) {
        return false;
    }
    if (!lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (!scalars.has_float_data()) {
        return false;
    }
    const auto& scalars_data = scalars.float_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

bool
operator==(const proto::schema::FieldData& lhs, const DoubleFieldData& rhs) {
    if (!IsEqual(lhs, reinterpret_cast<const Field&>(rhs))) {
        return false;
    }
    if (!lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (!scalars.has_double_data()) {
        return false;
    }
    const auto& scalars_data = scalars.double_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

bool
operator==(const proto::schema::FieldData& lhs, const VarCharFieldData& rhs) {
    if (!IsEqual(lhs, reinterpret_cast<const Field&>(rhs))) {
        return false;
    }
    if (!lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (!scalars.has_string_data()) {
        return false;
    }
    const auto& scalars_data = scalars.string_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }
    return std::equal(scalars_data.begin(), scalars_data.end(), rhs.Data().begin());
}

bool
operator==(const proto::schema::FieldData& lhs, const JSONFieldData& rhs) {
    if (!IsEqual(lhs, reinterpret_cast<const Field&>(rhs))) {
        return false;
    }
    if (!lhs.has_scalars()) {
        return false;
    }
    const auto& scalars = lhs.scalars();
    if (!scalars.has_json_data()) {
        return false;
    }

    const auto& scalars_data = scalars.json_data().data();
    if (scalars_data.size() != rhs.Count()) {
        return false;
    }

    EntityRows jsons;
    for (const auto& str : scalars_data) {
        jsons.emplace_back(std::move(nlohmann::json::parse(str)));
    }
    return std::equal(jsons.begin(), jsons.end(), rhs.Data().begin());
}

bool
operator==(const proto::schema::FieldData& lhs, const BinaryVecFieldData& rhs) {
    if (!IsEqual(lhs, reinterpret_cast<const Field&>(rhs))) {
        return false;
    }
    if (!lhs.has_vectors()) {
        return false;
    }

    const auto& vectors = lhs.vectors();
    if (!vectors.has_binary_vector()) {
        return false;
    }

    const auto& vectors_data = vectors.binary_vector();
    auto it = vectors_data.begin();
    const auto& strings = rhs.DataAsString();
    for (const auto& s : strings) {
        for (const auto ch : s) {
            if (it == vectors_data.end() || *it != ch) {
                return false;
            }
            ++it;
        }
    }

    return it == vectors_data.end();
}

bool
operator==(const proto::schema::FieldData& lhs, const FloatVecFieldData& rhs) {
    if (!IsEqual(lhs, reinterpret_cast<const Field&>(rhs))) {
        return false;
    }
    if (!lhs.has_vectors()) {
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

bool
operator==(const proto::schema::FieldData& lhs, const Field& rhs) {
    auto data_type = rhs.Type();
    switch (data_type) {
        case DataType::BOOL:
            return lhs == dynamic_cast<const BoolFieldData&>(rhs);
        case DataType::INT8:
            return lhs == dynamic_cast<const Int8FieldData&>(rhs);
        case DataType::INT16:
            return lhs == dynamic_cast<const Int16FieldData&>(rhs);
        case DataType::INT32:
            return lhs == dynamic_cast<const Int32FieldData&>(rhs);
        case DataType::INT64:
            return lhs == dynamic_cast<const Int64FieldData&>(rhs);
        case DataType::FLOAT:
            return lhs == dynamic_cast<const FloatFieldData&>(rhs);
        case DataType::DOUBLE:
            return lhs == dynamic_cast<const DoubleFieldData&>(rhs);
        case DataType::VARCHAR:
            return lhs == dynamic_cast<const VarCharFieldData&>(rhs);
        case DataType::JSON:
            return lhs == dynamic_cast<const JSONFieldData&>(rhs);
        case DataType::BINARY_VECTOR:
            return lhs == dynamic_cast<const BinaryVecFieldData&>(rhs);
        case DataType::FLOAT_VECTOR:
            return lhs == dynamic_cast<const FloatVecFieldData&>(rhs);
        case DataType::SPARSE_FLOAT_VECTOR:
            return lhs == dynamic_cast<const SparseFloatVecFieldData&>(rhs);
        default:
            return false;
    }
}

bool
operator==(const SegmentInfo& lhs, const SegmentInfo& rhs) {
    return lhs.CollectionID() == rhs.CollectionID() && lhs.PartitionID() == rhs.PartitionID() &&
           lhs.RowCount() == rhs.RowCount() && lhs.SegmentID() == rhs.SegmentID() && lhs.State() == rhs.State();
}

bool
operator==(const QuerySegmentInfo& lhs, const QuerySegmentInfo& rhs) {
    return lhs.CollectionID() == rhs.CollectionID() && lhs.PartitionID() == rhs.PartitionID() &&
           lhs.RowCount() == rhs.RowCount() && lhs.SegmentID() == rhs.SegmentID() && lhs.State() == rhs.State() &&
           lhs.IndexName() == rhs.IndexName() && lhs.IndexID() == rhs.IndexID() && lhs.NodeID() == rhs.NodeID();
}

}  // namespace milvus
