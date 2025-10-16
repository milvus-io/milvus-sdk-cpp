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
#include "milvus.pb.h"
#include "milvus/types/FieldData.h"
#include "milvus/types/HybridSearchArguments.h"
#include "milvus/types/QueryArguments.h"
#include "milvus/types/QueryResults.h"
#include "milvus/types/SearchArguments.h"
#include "milvus/types/SearchResults.h"
#include "schema.pb.h"

namespace milvus {
Status
CreateMilvusFieldData(const proto::schema::FieldData& proto_data, size_t offset, size_t count,
                      FieldDataPtr& field_data);

Status
CreateMilvusFieldData(const proto::schema::FieldData& proto_data, FieldDataPtr& field_data);

FieldDataPtr
CreateIDField(const std::string& name, const proto::schema::IDs& ids, size_t offset, size_t size);

FieldDataPtr
CreateScoreField(const std::string& name, const proto::schema::SearchResultData& data, size_t offset, size_t size);

Status
SetTargetVectors(const FieldDataPtr& vectors, milvus::proto::milvus::SearchRequest* rpc_request);

void
SetExtraParams(const std::unordered_map<std::string, std::string>& params,
               milvus::proto::milvus::SearchRequest* rpc_request);

Status
GetRowsFromFieldsData(const std::vector<FieldDataPtr>& fields, const std::set<std::string>& output_names,
                      EntityRows& rows);

Status
GetRowFromFieldsData(const std::vector<FieldDataPtr>& fields, size_t i, const std::set<std::string>& output_names,
                     EntityRow& row);

uint64_t
DeduceGuaranteeTimestamp(const ConsistencyLevel& level, const std::string& db_name, const std::string& collection_name);

Status
ConvertQueryRequest(const QueryArguments& arguments, const std::string& current_db,
                    proto::milvus::QueryRequest& rpc_request);

Status
ConvertQueryResults(const proto::milvus::QueryResults& rpc_results, QueryResults& results);

Status
ConvertSearchRequest(const SearchArguments& arguments, const std::string& current_db,
                     proto::milvus::SearchRequest& rpc_request);

Status
ConvertHybridSearchRequest(const HybridSearchArguments& arguments, const std::string& current_db,
                           proto::milvus::HybridSearchRequest& rpc_request);

Status
ConvertSearchResults(const proto::milvus::SearchResults& rpc_results, const std::string& pk_name,
                     SearchResults& results);

Status
CopyFieldData(const FieldDataPtr& src, uint64_t from, uint64_t to, FieldDataPtr& target);

Status
CopyFieldsData(const std::vector<FieldDataPtr>& src, uint64_t from, uint64_t to, std::vector<FieldDataPtr>& target);

Status
AppendFieldData(const FieldDataPtr& from, FieldDataPtr& to);

Status
AppendSearchResult(const SingleResult& from, SingleResult& to);

template <typename T>
Status
ParseParameter(const std::unordered_map<std::string, std::string>& params, const std::string& name, T& value) {
    auto iter = params.find(name);
    if (iter == params.end()) {
        return {StatusCode::INVALID_AGUMENT, "no such parameter"};
    }
    try {
        if (std::is_integral<T>::value) {
            value = static_cast<T>(std::stol(iter->second));
        } else if (std::is_floating_point<T>::value) {
            value = static_cast<T>(std::stof(iter->second));
        } else {
            return {StatusCode::INVALID_AGUMENT, "can only parse integer and float type value"};
        }
    } catch (...) {
        return {StatusCode::INVALID_AGUMENT, "parameter '" + name + "' value '" + iter->second + "' cannot be parsed"};
    }
    return Status::OK();
}

}  // namespace milvus
