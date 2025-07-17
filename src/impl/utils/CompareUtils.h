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
#include "milvus/types/FieldData.h"
#include "milvus/types/SegmentInfo.h"

namespace milvus {

bool
operator==(const proto::schema::FieldData& lhs, const BoolFieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const Int8FieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const Int16FieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const Int32FieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const Int64FieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const FloatFieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const DoubleFieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const VarCharFieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const JSONFieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const BinaryVecFieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const FloatVecFieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const proto::schema::FieldData& rhs);

bool
operator==(const proto::schema::FieldData& lhs, const Field& rhs);

bool
operator==(const SegmentInfo& lhs, const SegmentInfo& rhs);

bool
operator==(const QuerySegmentInfo& lhs, const QuerySegmentInfo& rhs);

/**
 * @brief To test two FieldData are equal
 */
template <typename T, DataType Dt>
bool
operator==(const FieldData<T, Dt>& lhs, const FieldData<T, Dt>& rhs) {
    return lhs.Name() == rhs.Name() && lhs.Count() == rhs.Count() && lhs.Data() == rhs.Data();
}

/**
 * @brief To test two FieldData are equal
 */
template <typename T, DataType Dt>
bool
operator==(const FieldData<T, Dt>& lhs, const Field& rhs) {
    return lhs == dynamic_cast<const FieldData<T, Dt>&>(rhs);
}

}  // namespace milvus
