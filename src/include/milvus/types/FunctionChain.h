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

#include <cstdint>
#include <milvus/thirdparty/nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Execution stages where a function chain can run.
 * Numeric values mirror schema.proto FunctionChainStage.
 */
enum class FunctionChainStage {
    UNSPECIFIED = 0,
    INGESTION = 1,
    PRE_PROCESS = 2,
    L0_RERANK = 3,
    L1_RERANK = 4,
    L2_RERANK = 5,
    POST_PROCESS = 6,
};

/**
 * @brief A reference to a collection field used as a function-chain argument.
 */
class MILVUS_SDK_API FunctionChainColumnRef {
 public:
    FunctionChainColumnRef() = default;
    explicit FunctionChainColumnRef(std::string name);

    const std::string&
    Name() const;

    void
    SetName(std::string name);

 private:
    std::string name_;
};

/**
 * @brief Create a column reference for use in a function chain expression.
 */
MILVUS_SDK_API FunctionChainColumnRef
col(const std::string& name);

/**
 * @brief A single argument of a function-chain expression: either a column reference or a literal.
 */
class MILVUS_SDK_API FunctionChainExprArg {
 public:
    FunctionChainExprArg() = default;
    explicit FunctionChainExprArg(FunctionChainColumnRef column);
    explicit FunctionChainExprArg(nlohmann::json literal);

    bool
    IsColumn() const;

    bool
    IsLiteral() const;

    const std::string&
    ColumnName() const;

    const nlohmann::json&
    Literal() const;

 private:
    bool is_column_{false};
    std::string column_name_;
    nlohmann::json literal_;
};

/**
 * @brief Function invocation expression used by a function chain operation.
 * e.g. "num_combine", "decay", "round_decimal", etc.
 */
class MILVUS_SDK_API FunctionChainExpr {
 public:
    FunctionChainExpr() = default;
    explicit FunctionChainExpr(std::string name);

    FunctionChainExpr&
    AddColumnArg(const std::string& column);

    FunctionChainExpr&
    AddLiteralArg(const nlohmann::json& literal);

    FunctionChainExpr&
    AddParam(const std::string& key, const nlohmann::json& value);

    const std::string&
    Name() const;

    const std::vector<FunctionChainExprArg>&
    Args() const;

    const std::unordered_map<std::string, nlohmann::json>&
    Params() const;

 private:
    std::string name_;
    std::vector<FunctionChainExprArg> args_;
    std::unordered_map<std::string, nlohmann::json> params_;
};

/**
 * @brief A single operation in a function chain pipeline, such as "map", "sort", or "limit".
 */
class MILVUS_SDK_API FunctionChainOp {
 public:
    FunctionChainOp() = default;
    explicit FunctionChainOp(std::string op);

    FunctionChainOp&
    WithExpr(const FunctionChainExpr& expr);

    FunctionChainOp&
    AddInput(const std::string& input);

    FunctionChainOp&
    AddOutput(const std::string& output);

    FunctionChainOp&
    AddParam(const std::string& key, const nlohmann::json& value);

    const std::string&
    Op() const;

    bool
    HasExpr() const;

    const FunctionChainExpr&
    Expr() const;

    const std::vector<std::string>&
    Inputs() const;

    const std::vector<std::string>&
    Outputs() const;

    const std::unordered_map<std::string, nlohmann::json>&
    Params() const;

 private:
    std::string op_;
    bool has_expr_{false};
    FunctionChainExpr expr_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::unordered_map<std::string, nlohmann::json> params_;
};

/**
 * @brief Builder for composing and serializing a function chain for search.
 */
class MILVUS_SDK_API FunctionChain {
 public:
    FunctionChain() = default;
    explicit FunctionChain(FunctionChainStage stage, std::string name = "");

    FunctionChain&
    WithName(std::string name);

    /**
     * @brief Append a map operation that writes an expression result to an output field.
     */
    FunctionChain&
    Map(const std::string& output, const FunctionChainExpr& expr);

    /**
     * @brief Append a sort operation by column, optionally with a tie-break column.
     */
    FunctionChain&
    Sort(const std::string& by, bool desc = true, const std::string& tie_break_col = "");

    /**
     * @brief Append a limit operation with an optional offset.
     */
    FunctionChain&
    Limit(int64_t limit, int64_t offset = 0);

    /**
     * @brief Append a raw operation.
     */
    FunctionChain&
    AddOp(const FunctionChainOp& op);

    FunctionChainStage
    Stage() const;

    const std::string&
    Name() const;

    const std::vector<FunctionChainOp>&
    Ops() const;

 private:
    FunctionChainStage stage_{FunctionChainStage::UNSPECIFIED};
    std::string name_;
    std::vector<FunctionChainOp> ops_;
};

}  // namespace milvus
