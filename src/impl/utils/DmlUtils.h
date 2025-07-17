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

#include "common.pb.h"
#include "milvus/types/CollectionDesc.h"
#include "milvus/types/ConsistencyLevel.h"
#include "milvus/types/FieldData.h"
#include "milvus/types/FieldSchema.h"

namespace milvus {

bool
IsInputField(const FieldSchema& field_schema, bool is_upsert);

Status
CheckInsertInput(const CollectionDescPtr& collection_desc, const std::vector<FieldDataPtr>& fields, bool is_upsert);

bool
IsRealFailure(const proto::common::Status& status);

uint64_t
DeduceGuaranteeTimestamp(const ConsistencyLevel& level, const std::string& db_name, const std::string& collection_name);

}  // namespace milvus
