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

#include <string>
#include <vector>

#include "milvus/Export.h"
#include "milvus/types/RowPolicyAction.h"
#include "milvus/types/RowPolicyType.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::CreateRowPolicy() and UpdateRowPolicy().
 */
class MILVUS_SDK_API RowPolicyRequest {
 public:
    /**
     * @brief Constructor
     */
    RowPolicyRequest() = default;

    /**
     * @brief Get database name.
     * @return the database name.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name.
     * @param [in] db_name the database name.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set database name.
     * @param [in] db_name the database name.
     */
    RowPolicyRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Get collection name.
     * @return the collection name.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set collection name.
     * @param [in] collection_name the collection name.
     */
    void
    SetCollectionName(const std::string& collection_name);

    /**
     * @brief Set collection name.
     * @param [in] collection_name the collection name.
     */
    RowPolicyRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Get policy name.
     * @return the policy name.
     */
    const std::string&
    PolicyName() const;

    /**
     * @brief Set policy name.
     * @param [in] policy_name the policy name.
     */
    void
    SetPolicyName(const std::string& policy_name);

    /**
     * @brief Set policy name.
     * @param [in] policy_name the policy name.
     */
    RowPolicyRequest&
    WithPolicyName(const std::string& policy_name);

    /**
     * @brief Get policy type.
     * @return the policy type.
     */
    RowPolicyType
    PolicyType() const;

    /**
     * @brief Set policy type.
     * @param [in] policy_type the policy type.
     */
    void
    SetPolicyType(RowPolicyType policy_type);

    /**
     * @brief Set policy type.
     * @param [in] policy_type the policy type.
     */
    RowPolicyRequest&
    WithPolicyType(RowPolicyType policy_type);

    /**
     * @brief Get the row-bearing actions the policy applies to.
     * @return the actions.
     */
    const std::vector<RowPolicyAction>&
    Actions() const;

    /**
     * @brief Set the row-bearing actions the policy applies to.
     * @param [in] actions the actions.
     */
    void
    SetActions(std::vector<RowPolicyAction>&& actions);

    /**
     * @brief Set the row-bearing actions the policy applies to.
     * @param [in] actions the actions.
     */
    RowPolicyRequest&
    WithActions(std::vector<RowPolicyAction>&& actions);

    /**
     * @brief Get the filter expression restricting which rows are visible.
     * @return the using expression.
     */
    const std::string&
    UsingExpr() const;

    /**
     * @brief Set the filter expression restricting which rows are visible.
     * @param [in] expr the using expression.
     */
    void
    SetUsingExpr(const std::string& expr);

    /**
     * @brief Set the filter expression restricting which rows are visible.
     * @param [in] expr the using expression.
     */
    RowPolicyRequest&
    WithUsingExpr(const std::string& expr);

    /**
     * @brief Get the expression restricting which rows may be written.
     * @return the check expression.
     */
    const std::string&
    CheckExpr() const;

    /**
     * @brief Set the expression restricting which rows may be written.
     * @param [in] expr the check expression.
     */
    void
    SetCheckExpr(const std::string& expr);

    /**
     * @brief Set the expression restricting which rows may be written.
     * @param [in] expr the check expression.
     */
    RowPolicyRequest&
    WithCheckExpr(const std::string& expr);

    /**
     * @brief Get description of the policy.
     * @return the description.
     */
    const std::string&
    Description() const;

    /**
     * @brief Set description of the policy.
     * @param [in] description the description.
     */
    void
    SetDescription(const std::string& description);

    /**
     * @brief Set description of the policy.
     * @param [in] description the description.
     */
    RowPolicyRequest&
    WithDescription(const std::string& description);

 protected:
    std::string db_name_;
    std::string collection_name_;
    std::string policy_name_;
    RowPolicyType policy_type_{RowPolicyType::UNKNOWN};
    std::vector<RowPolicyAction> actions_;
    std::string using_expr_;
    std::string check_expr_;
    std::string description_;
};

using CreateRowPolicyRequest = RowPolicyRequest;
using UpdateRowPolicyRequest = RowPolicyRequest;

}  // namespace milvus
