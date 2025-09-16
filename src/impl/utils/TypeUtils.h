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

#include <unordered_map>

#include "milvus.pb.h"
#include "milvus/types/CollectionSchema.h"
#include "milvus/types/ConsistencyLevel.h"
#include "milvus/types/FieldData.h"
#include "milvus/types/IDArray.h"
#include "milvus/types/IndexState.h"
#include "milvus/types/IndexType.h"
#include "milvus/types/MetricType.h"
#include "milvus/types/ResourceGroupConfig.h"
#include "milvus/types/SearchResults.h"
#include "milvus/types/SegmentInfo.h"

namespace milvus {

proto::schema::DataType
DataTypeCast(DataType type);

DataType
DataTypeCast(proto::schema::DataType type);

MetricType
MetricTypeCast(const std::string& type);

IndexType
IndexTypeCast(const std::string& type);

void
ConvertFieldSchema(const proto::schema::FieldSchema& proto_schema, FieldSchema& schema);

void
ConvertCollectionSchema(const proto::schema::CollectionSchema& proto_schema, CollectionSchema& schema);

void
ConvertFieldSchema(const FieldSchema& schema, proto::schema::FieldSchema& proto_schema);

void
ConvertCollectionSchema(const CollectionSchema& schema, proto::schema::CollectionSchema& proto_schema);

SegmentState
SegmentStateCast(proto::common::SegmentState state);

proto::common::SegmentState
SegmentStateCast(SegmentState state);

IndexStateCode
IndexStateCast(proto::common::IndexState state);

bool
IsVectorType(DataType type);

std::string
Base64Encode(const std::string& val);

proto::common::ConsistencyLevel
ConsistencyLevelCast(const ConsistencyLevel& level);

ConsistencyLevel
ConsistencyLevelCast(const proto::common::ConsistencyLevel& level);

void
ConvertResourceGroupConfig(const ResourceGroupConfig& config, proto::rg::ResourceGroupConfig* rpc_config);

void
ConvertResourceGroupConfig(const proto::rg::ResourceGroupConfig& rpc_config, ResourceGroupConfig& config);

std::string
doubleToString(double val);

}  // namespace milvus
