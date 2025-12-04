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

#include "./FieldDataSchema.h"
#include "common.pb.h"
#include "milvus.pb.h"
#include "milvus/types/CollectionDesc.h"
#include "milvus/types/ConsistencyLevel.h"
#include "milvus/types/IDArray.h"
#include "schema.pb.h"

namespace milvus {

bool
IsInputField(const FieldSchema& field_schema, bool is_upsert);

Status
CheckInsertInput(const CollectionDescPtr& collection_desc, const std::vector<FieldDataPtr>& fields, bool is_upsert);

bool
IsRealFailure(const proto::common::Status& status);

std::string
EncodeSparseFloatVector(const SparseFloatVecFieldData::ElementT& sparse);

Status
ParseSparseFloatVector(const nlohmann::json& obj, const std::string& field_name, std::map<uint32_t, float>& pairs);

Status
CreateProtoFieldData(const FieldDataSchema& data_schema, proto::schema::FieldData& field_data);

Status
CreateProtoFieldDatas(const CollectionSchema& collection_schema, const std::vector<FieldDataPtr>& fields,
                      std::vector<proto::schema::FieldData>& rpc_fields);

IDArray
CreateIDArray(const proto::schema::IDs& ids);

Status
CheckAndSetBinaryVector(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::VectorField* vf);

Status
CheckAndSetFloatVector(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::VectorField* vf);

Status
CheckAndSetSparseFloatVector(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::VectorField* vf);

Status
CheckAndSetFloat16Vector(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::VectorField* vf);

Status
CheckAndSetArray(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::ArrayArray* aa);

Status
CheckAndSetNullableDefaultScalar(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::FieldData& fd);

Status
CheckAndSetScalar(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::ScalarField* sf, bool is_array);

Status
CheckAndSetFieldValue(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::FieldData& fd);

Status
CheckAndSetRowData(const EntityRows& rows, const CollectionSchema& schema, bool is_upsert,
                   std::vector<proto::schema::FieldData>& rpc_fields);

}  // namespace milvus
