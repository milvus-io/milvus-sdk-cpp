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

#include "milvus/request/dql/HybridSearchRequest.h"

#include <memory>

#include "../../utils/ExtraParamUtils.h"

namespace milvus {

const std::vector<SubSearchRequestPtr>&
HybridSearchRequest::SubRequests() const {
    return sub_requests_;
}

void
HybridSearchRequest::SetSubRequests(std::vector<SubSearchRequestPtr>&& requests) {
    sub_requests_ = std::move(requests);
}

HybridSearchRequest&
HybridSearchRequest::WithSubRequests(std::vector<SubSearchRequestPtr>&& requests) {
    sub_requests_ = std::move(requests);
    return *this;
}

HybridSearchRequest&
HybridSearchRequest::AddSubRequest(const SubSearchRequestPtr& request) {
    sub_requests_.emplace_back(request);
    return *this;
}

FunctionPtr
HybridSearchRequest::Rerank() const {
    return function_;
}

Status
HybridSearchRequest::SetRerank(const FunctionPtr& rerank) {
    function_ = rerank;
    return Status::OK();
}

HybridSearchRequest&
HybridSearchRequest::WithRerank(const FunctionPtr& rerank) {
    function_ = rerank;
    return *this;
}

const std::vector<FunctionChain>&
HybridSearchRequest::FunctionChains() const {
    return function_chains_;
}

void
HybridSearchRequest::SetFunctionChains(std::vector<FunctionChain>&& function_chains) {
    function_chains_ = std::move(function_chains);
}

HybridSearchRequest&
HybridSearchRequest::WithFunctionChains(std::vector<FunctionChain>&& function_chains) {
    SetFunctionChains(std::move(function_chains));
    return *this;
}

HybridSearchRequest&
HybridSearchRequest::AddFunctionChain(const FunctionChain& function_chain) {
    function_chains_.push_back(function_chain);
    return *this;
}

int64_t
HybridSearchRequest::Limit() const {
    return limit_;
}

Status
HybridSearchRequest::SetLimit(int64_t limit) {
    limit_ = limit;
    return Status::OK();
}

HybridSearchRequest&
HybridSearchRequest::WithLimit(int64_t limit) {
    limit_ = limit;
    return *this;
}

int64_t
HybridSearchRequest::Offset() const {
    return GetExtraInt64(extra_params_, "offset", 0);
}

void
HybridSearchRequest::SetOffset(int64_t offset) {
    SetExtraInt64(extra_params_, "offset", offset);
}

HybridSearchRequest&
HybridSearchRequest::WithOffset(int64_t offset) {
    SetOffset(offset);
    return *this;
}

int64_t
HybridSearchRequest::GetRoundDecimal() const {
    return GetExtraInt64(extra_params_, "round_decimal", -1);
}

void
HybridSearchRequest::SetRoundDecimal(int64_t round_decimal) {
    SetExtraInt64(extra_params_, "round_decimal", round_decimal);
}

HybridSearchRequest&
HybridSearchRequest::WithRoundDecimal(int64_t round_decimal) {
    SetRoundDecimal(round_decimal);
    return *this;
}

bool
HybridSearchRequest::IgnoreGrowing() const {
    return GetExtraBool(extra_params_, "ignore_growing", false);
}

void
HybridSearchRequest::SetIgnoreGrowing(bool ignore_growing) {
    SetExtraBool(extra_params_, "ignore_growing", ignore_growing);
}

HybridSearchRequest&
HybridSearchRequest::WithIgnoreGrowing(bool ignore_growing) {
    SetIgnoreGrowing(ignore_growing);
    return *this;
}

HybridSearchRequest&
HybridSearchRequest::AddExtraParam(const std::string& key, const std::string& value) {
    extra_params_[key] = value;
    return *this;
}

const std::unordered_map<std::string, std::string>&
HybridSearchRequest::ExtraParams() const {
    return extra_params_;
}

std::string
HybridSearchRequest::GetGroupByField() const {
    return GetExtraStr(extra_params_, "group_by_field", "");
}

void
HybridSearchRequest::SetGroupByField(const std::string& field_name) {
    SetExtraStr(extra_params_, "group_by_field", field_name);
}

HybridSearchRequest&
HybridSearchRequest::WithGroupByField(const std::string& field_name) {
    SetGroupByField(field_name);
    return *this;
}

int64_t
HybridSearchRequest::GroupSize() const {
    return GetExtraInt64(extra_params_, "group_size", 1);
}

void
HybridSearchRequest::SetGroupSize(int64_t group_size) {
    SetExtraInt64(extra_params_, "group_size", group_size);
}

HybridSearchRequest&
HybridSearchRequest::WithGroupSize(int64_t group_size) {
    SetGroupSize(group_size);
    return *this;
}

bool
HybridSearchRequest::StrictGroupSize() const {
    return GetExtraBool(extra_params_, "strict_group_size", false);
}

void
HybridSearchRequest::SetStrictGroupSize(bool strict_group_size) {
    SetExtraBool(extra_params_, "strict_group_size", strict_group_size);
}

HybridSearchRequest&
HybridSearchRequest::WithStrictGroupSize(bool strict_group_size) {
    SetStrictGroupSize(strict_group_size);
    return *this;
}

Status
HybridSearchRequest::Validate() const {
    auto status = ValidateLimit(limit_);
    if (!status.IsOk()) {
        return status;
    }

    status = ValidateRoundDecimal(extra_params_);
    if (!status.IsOk()) {
        return status;
    }

    for (const auto& sub_request : sub_requests_) {
        if (sub_request == nullptr) {
            return {StatusCode::INVALID_ARGUMENT, "Sub request can not be null!"};
        }
        auto status = sub_request->Validate();
        if (!status.IsOk()) {
            return status;
        }
    }
    if (function_ != nullptr && !function_chains_.empty()) {
        return {StatusCode::INVALID_ARGUMENT, "Function chains and rerank cannot be used together"};
    }
    if (function_ == nullptr && function_chains_.empty()) {
        return {StatusCode::INVALID_ARGUMENT, "Rerank function or function chains is undefined!"};
    }
    if (function_ != nullptr) {
        if (function_->GetFunctionType() != FunctionType::RERANK) {
            return {StatusCode::INVALID_ARGUMENT, "Hybrid search only accepts RERANK function!"};
        }
    }
    for (const auto& chain : function_chains_) {
        if (chain.Stage() == FunctionChainStage::UNSPECIFIED) {
            return {StatusCode::INVALID_ARGUMENT, "UNSPECIFIED function chain stage is not supported for search"};
        }
        for (const auto& op : chain.Ops()) {
            if (op.Op().empty()) {
                return {StatusCode::INVALID_ARGUMENT, "Function chain op name cannot be empty"};
            }
            for (const auto& input : op.Inputs()) {
                if (input.empty()) {
                    return {StatusCode::INVALID_ARGUMENT, "Function chain op input column name cannot be empty"};
                }
            }
            for (const auto& output : op.Outputs()) {
                if (output.empty()) {
                    return {StatusCode::INVALID_ARGUMENT, "Function chain op output column name cannot be empty"};
                }
            }
            if (op.HasExpr()) {
                const auto& expr = op.Expr();
                if (expr.Name().empty()) {
                    return {StatusCode::INVALID_ARGUMENT, "Function chain expression name cannot be empty"};
                }
                for (const auto& arg : expr.Args()) {
                    if (arg.IsColumn() && arg.ColumnName().empty()) {
                        return {StatusCode::INVALID_ARGUMENT, "Function chain expression column name cannot be empty"};
                    }
                }
            }
        }
    }

    return Status::OK();
}

}  // namespace milvus
