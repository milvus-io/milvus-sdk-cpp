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

#include "milvus/types/FunctionChain.h"

namespace milvus {

FunctionChainColumnRef::FunctionChainColumnRef(std::string name) : name_(std::move(name)) {
}

const std::string&
FunctionChainColumnRef::Name() const {
    return name_;
}

void
FunctionChainColumnRef::SetName(std::string name) {
    name_ = std::move(name);
}

FunctionChainColumnRef
col(const std::string& name) {
    return FunctionChainColumnRef(name);
}

FunctionChainExprArg::FunctionChainExprArg(FunctionChainColumnRef column)
    : is_column_(true), column_name_(column.Name()) {
}

FunctionChainExprArg::FunctionChainExprArg(nlohmann::json literal) : literal_(std::move(literal)) {
}

bool
FunctionChainExprArg::IsColumn() const {
    return is_column_;
}

bool
FunctionChainExprArg::IsLiteral() const {
    return !is_column_;
}

const std::string&
FunctionChainExprArg::ColumnName() const {
    return column_name_;
}

const nlohmann::json&
FunctionChainExprArg::Literal() const {
    return literal_;
}

FunctionChainExpr::FunctionChainExpr(std::string name) : name_(std::move(name)) {
}

FunctionChainExpr&
FunctionChainExpr::AddColumnArg(const std::string& column) {
    args_.emplace_back(col(column));
    return *this;
}

FunctionChainExpr&
FunctionChainExpr::AddLiteralArg(const nlohmann::json& literal) {
    args_.emplace_back(literal);
    return *this;
}

FunctionChainExpr&
FunctionChainExpr::AddParam(const std::string& key, const nlohmann::json& value) {
    params_[key] = value;
    return *this;
}

const std::string&
FunctionChainExpr::Name() const {
    return name_;
}

const std::vector<FunctionChainExprArg>&
FunctionChainExpr::Args() const {
    return args_;
}

const std::unordered_map<std::string, nlohmann::json>&
FunctionChainExpr::Params() const {
    return params_;
}

FunctionChainOp::FunctionChainOp(std::string op) : op_(std::move(op)) {
}

FunctionChainOp&
FunctionChainOp::WithExpr(const FunctionChainExpr& expr) {
    expr_ = expr;
    has_expr_ = true;
    return *this;
}

FunctionChainOp&
FunctionChainOp::AddInput(const std::string& input) {
    inputs_.push_back(input);
    return *this;
}

FunctionChainOp&
FunctionChainOp::AddOutput(const std::string& output) {
    outputs_.push_back(output);
    return *this;
}

FunctionChainOp&
FunctionChainOp::AddParam(const std::string& key, const nlohmann::json& value) {
    params_[key] = value;
    return *this;
}

const std::string&
FunctionChainOp::Op() const {
    return op_;
}

bool
FunctionChainOp::HasExpr() const {
    return has_expr_;
}

const FunctionChainExpr&
FunctionChainOp::Expr() const {
    return expr_;
}

const std::vector<std::string>&
FunctionChainOp::Inputs() const {
    return inputs_;
}

const std::vector<std::string>&
FunctionChainOp::Outputs() const {
    return outputs_;
}

const std::unordered_map<std::string, nlohmann::json>&
FunctionChainOp::Params() const {
    return params_;
}

FunctionChain::FunctionChain(FunctionChainStage stage, std::string name) : stage_(stage), name_(std::move(name)) {
}

FunctionChain&
FunctionChain::WithName(std::string name) {
    name_ = std::move(name);
    return *this;
}

FunctionChain&
FunctionChain::Map(const std::string& output, const FunctionChainExpr& expr) {
    FunctionChainOp op("map");
    op.WithExpr(expr).AddOutput(output);
    ops_.push_back(std::move(op));
    return *this;
}

FunctionChain&
FunctionChain::Sort(const std::string& by, bool desc, const std::string& tie_break_col) {
    FunctionChainOp op("sort");
    op.AddInput(by).AddParam("column", by).AddParam("desc", desc);
    if (!tie_break_col.empty()) {
        op.AddInput(tie_break_col).AddParam("tie_break_col", tie_break_col);
    }
    ops_.push_back(std::move(op));
    return *this;
}

FunctionChain&
FunctionChain::Limit(int64_t limit, int64_t offset) {
    FunctionChainOp op("limit");
    op.AddParam("limit", limit).AddParam("offset", offset);
    ops_.push_back(std::move(op));
    return *this;
}

FunctionChain&
FunctionChain::AddOp(const FunctionChainOp& op) {
    ops_.push_back(op);
    return *this;
}

FunctionChainStage
FunctionChain::Stage() const {
    return stage_;
}

const std::string&
FunctionChain::Name() const {
    return name_;
}

const std::vector<FunctionChainOp>&
FunctionChain::Ops() const {
    return ops_;
}

}  // namespace milvus
