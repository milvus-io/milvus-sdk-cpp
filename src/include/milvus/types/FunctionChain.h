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
    /**
     * @brief Constructor
     */
    FunctionChainColumnRef() = default;

    /**
     * @brief Construct a column reference.
     *
     * @param [in] name field name, e.g. "$score".
     */
    explicit FunctionChainColumnRef(std::string name);

    /**
     * @brief Get the referenced field name.
     * @return the name.
     */
    const std::string&
    Name() const;

    /**
     * @brief Set the referenced field name.
     * @param [in] name the name.
     */
    void
    SetName(std::string name);

 private:
    std::string name_;
};

/**
 * @brief Create a column reference for use in a function chain expression.
 * @param [in] name the name.
 */
MILVUS_SDK_API FunctionChainColumnRef
col(const std::string& name);

/**
 * @brief A single argument of a function-chain expression: either a column reference or a literal.
 */
class MILVUS_SDK_API FunctionChainExprArg {
 public:
    /**
     * @brief Constructor
     */
    FunctionChainExprArg() = default;

    /**
     * @brief Construct a column-reference argument.
     * @param [in] column the column.
     */
    explicit FunctionChainExprArg(FunctionChainColumnRef column);

    /**
     * @brief Construct a literal argument.
     * @param [in] literal the literal.
     */
    explicit FunctionChainExprArg(nlohmann::json literal);

    /**
     * @brief Whether this argument is a column reference.
     * @return true if this is a column reference.
     */
    bool
    IsColumn() const;

    /**
     * @brief Whether this argument is a literal value.
     * @return true if this is a literal.
     */
    bool
    IsLiteral() const;

    /**
     * @brief Get the column name when this argument is a column reference.
     * @return the column name.
     */
    const std::string&
    ColumnName() const;

    /**
     * @brief Get the literal value when this argument is a literal.
     * @return the literal.
     */
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
    /**
     * @brief Constructor
     */
    FunctionChainExpr() = default;

    /**
     * @brief Construct a function expression.
     *
     * @param [in] name expression name, e.g. "num_combine", "decay", "round_decimal".
     */
    explicit FunctionChainExpr(std::string name);

    /**
     * @brief Append a column-reference argument to the expression.
     * @param [in] column the column.
     */
    FunctionChainExpr&
    AddColumnArg(const std::string& column);

    /**
     * @brief Append a literal argument to the expression.
     * @param [in] literal the literal.
     */
    FunctionChainExpr&
    AddLiteralArg(const nlohmann::json& literal);

    /**
     * @brief Set a named parameter of the expression.
     * @param [in] key the key.
     * @param [in] value the value.
     */
    FunctionChainExpr&
    AddParam(const std::string& key, const nlohmann::json& value);

    /**
     * @brief Get the expression name.
     * @return the name.
     */
    const std::string&
    Name() const;

    /**
     * @brief Get the expression arguments.
     * @return the args.
     */
    const std::vector<FunctionChainExprArg>&
    Args() const;

    /**
     * @brief Get the named parameters of the expression.
     * @return the params.
     */
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
    /**
     * @brief Constructor
     */
    FunctionChainOp() = default;

    /**
     * @brief Construct a function-chain operation.
     *
     * @param [in] op operation name, e.g. "map", "sort", "limit".
     */
    explicit FunctionChainOp(std::string op);

    /**
     * @brief Attach a function expression to this operation.
     * @param [in] expr the expr.
     */
    FunctionChainOp&
    WithExpr(const FunctionChainExpr& expr);

    /**
     * @brief Add an input column to this operation.
     * @param [in] input the input.
     */
    FunctionChainOp&
    AddInput(const std::string& input);

    /**
     * @brief Add an output column to this operation.
     * @param [in] output the output.
     */
    FunctionChainOp&
    AddOutput(const std::string& output);

    /**
     * @brief Set a named parameter of this operation.
     * @param [in] key the key.
     * @param [in] value the value.
     */
    FunctionChainOp&
    AddParam(const std::string& key, const nlohmann::json& value);

    /**
     * @brief Get the operation name.
     * @return the op.
     */
    const std::string&
    Op() const;

    /**
     * @brief Whether this operation carries a function expression.
     * @return true if an expression is attached.
     */
    bool
    HasExpr() const;

    /**
     * @brief Get the attached function expression.
     * @return the expr.
     */
    const FunctionChainExpr&
    Expr() const;

    /**
     * @brief Get the input columns of this operation.
     * @return the inputs.
     */
    const std::vector<std::string>&
    Inputs() const;

    /**
     * @brief Get the output columns of this operation.
     * @return the outputs.
     */
    const std::vector<std::string>&
    Outputs() const;

    /**
     * @brief Get the named parameters of this operation.
     * @return the params.
     */
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
    /**
     * @brief Constructor
     */
    FunctionChain() = default;

    /**
     * @brief Construct a function chain.
     *
     * @param [in] stage execution stage of the chain.
     * @param [in] name chain name.
     */
    explicit FunctionChain(FunctionChainStage stage, std::string name = "");

    /**
     * @brief Set the chain name.
     * @param [in] name the name.
     */
    FunctionChain&
    WithName(std::string name);

    /**
     * @brief Append a map operation that writes an expression result to an output field.
     * @param [in] output the output.
     * @param [in] expr the expr.
     */
    FunctionChain&
    Map(const std::string& output, const FunctionChainExpr& expr);

    /**
     * @brief Append a sort operation by column, optionally with a tie-break column.
     * @param [in] by the by.
     * @param [in] desc the desc.
     * @param [in] tie_break_col the tie break col.
     */
    FunctionChain&
    Sort(const std::string& by, bool desc = true, const std::string& tie_break_col = "");

    /**
     * @brief Append a limit operation with an optional offset.
     * @param [in] limit the limit.
     * @param [in] offset the offset.
     */
    FunctionChain&
    Limit(int64_t limit, int64_t offset = 0);

    /**
     * @brief Append a raw operation.
     * @param [in] op the op.
     */
    FunctionChain&
    AddOp(const FunctionChainOp& op);

    /**
     * @brief Get the execution stage of the chain.
     * @return the stage.
     */
    FunctionChainStage
    Stage() const;

    /**
     * @brief Get the chain name.
     * @return the name.
     */
    const std::string&
    Name() const;

    /**
     * @brief Get the operations of the chain in pipeline order.
     * @return the ops.
     */
    const std::vector<FunctionChainOp>&
    Ops() const;

 private:
    FunctionChainStage stage_{FunctionChainStage::UNSPECIFIED};
    std::string name_;
    std::vector<FunctionChainOp> ops_;
};

}  // namespace milvus
