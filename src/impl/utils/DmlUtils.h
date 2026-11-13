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
#include "milvus/types/FieldPartialUpdateOp.h"
#include "milvus/types/IDArray.h"
#include "schema.pb.h"

namespace milvus {
class DmlResults;

bool
IsInputField(const FieldSchema& field_schema, bool is_upsert);

Status
CheckInsertInput(const CollectionDescPtr& collection_desc, const std::vector<FieldDataPtr>& fields, bool is_upsert,
                 bool partial_upsert);

bool
IsRealFailure(const proto::common::Status& status);

void
FillDmlCost(const proto::common::Status& status, DmlResults& results);

std::string
EncodeSparseFloatVector(const SparseFloatVecFieldData::ElementT& sparse);

Status
ParseSparseFloatVector(const nlohmann::json& obj, const std::string& field_name, std::map<uint32_t, float>& pairs);

Status
FillStructProtoFields(const std::vector<nlohmann::json>& dict_list, const StructFieldSchema& struct_schema,
                      const std::set<std::string>& output_fields,
                      std::map<std::string, proto::schema::FieldData>& proto_sub_fields,
                      const std::set<std::string>* sub_field_whitelist = nullptr);

/**
 * @brief Build the per-struct sub-field whitelist from PATH_REPLACE field ops.
 *
 * A child path ([index][child]) selects only the targeted child column, while a whole
 * element path ([index]) keeps every sub-field. When any whole-element path targets a
 * struct, the whitelist stays empty ("keep all") independent of op order. A PATH_REPLACE
 * op targeting a non-struct field is rejected. The path index itself is validated by the
 * server.
 * @param [in] field_ops the upsert field ops (nullable).
 * @param [in] struct_field_names the collection's struct field names (nullable).
 * @param [out] whitelist the per-struct sub-field whitelist.
 * @return the validation status.
 */
Status
BuildStructSubFieldWhitelist(const std::vector<FieldPartialUpdateOp>* field_ops,
                             const std::set<std::string>* struct_field_names,
                             std::map<std::string, std::set<std::string>>& whitelist);

/**
 * @brief Append one null struct row to every processed sub-field of a nullable struct.
 *
 * Writes the shared row mask (false) on each sub-field's FieldData and payload, without
 * appending a data cell, matching the pymilvus/Java wire layout for a whole null struct row.
 * @param [in] struct_schema the struct field schema.
 * @param [in] output_fields the output field names to skip.
 * @param [in,out] proto_sub_fields the accumulated sub-field proto data.
 * @return the validation status.
 */
Status
AppendStructNullRow(const StructFieldSchema& struct_schema, const std::set<std::string>& output_fields,
                    std::map<std::string, proto::schema::FieldData>& proto_sub_fields);

/**
 * @brief Validate that every PATH_REPLACE child in the whitelist exists as a sub-field.
 *
 * A typo like [0][bogus] would otherwise pass the operand key check and reach the server
 * with an empty struct payload; pymilvus rejects it client-side ("PATH_REPLACE child
 * 'bogus' not found in 'clips'").
 * @param [in] struct_fields the collection's struct field schemas.
 * @param [in] whitelist the per-struct sub-field whitelist built from PATH_REPLACE ops.
 * @return the validation status.
 */
Status
ValidateStructSubFieldWhitelist(const std::vector<StructFieldSchema>& struct_fields,
                                const std::map<std::string, std::set<std::string>>& whitelist);

Status
CreateProtoFieldData(const FieldDataSchema& data_schema, proto::schema::FieldData& field_data);

Status
CreateProtoFieldDatas(const CollectionSchema& collection_schema, const std::vector<FieldDataPtr>& fields,
                      std::vector<proto::schema::FieldData>& rpc_fields,
                      const std::vector<FieldPartialUpdateOp>* field_ops = nullptr);

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
CheckAndSetInt8Vector(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::VectorField* vf);

Status
CheckAndSetArray(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::ArrayArray* aa);

Status
CheckAndSetNullableDefaultScalar(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::FieldData& fd);

Status
CheckAndSetScalar(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::ScalarField* sf, bool is_array);

Status
CheckAndSetFieldValue(const nlohmann::json& obj, const FieldSchema& fs, proto::schema::FieldData& fd);

Status
CheckAndSetRowData(const EntityRows& rows, const CollectionSchema& schema, bool is_upsert, bool partial_upsert,
                   std::vector<proto::schema::FieldData>& rpc_fields,
                   const std::vector<FieldPartialUpdateOp>* field_ops = nullptr);

}  // namespace milvus
