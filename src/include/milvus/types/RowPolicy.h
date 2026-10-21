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
#include <string>
#include <vector>

#include "milvus/Export.h"
#include "milvus/types/RowPolicyAction.h"
#include "milvus/types/RowPolicyType.h"

namespace milvus {

/**
 * @brief Definition of one row policy. Returned by MilvusClientV2::ListRowPolicies().
 */
class MILVUS_SDK_API RowPolicy {
 public:
    /**
     * @brief Name of the policy.
     * @return the policy name.
     */
    const std::string&
    PolicyName() const;

    /**
     * @brief Set name of the policy.
     * @param [in] name the policy name.
     */
    void
    SetPolicyName(const std::string& name);

    /**
     * @brief Type of the policy.
     * @return the policy type.
     */
    RowPolicyType
    PolicyType() const;

    /**
     * @brief Set type of the policy.
     * @param [in] policy_type the policy type.
     */
    void
    SetPolicyType(RowPolicyType policy_type);

    /**
     * @brief Row-bearing actions the policy applies to.
     * @return the actions.
     */
    const std::vector<RowPolicyAction>&
    Actions() const;

    /**
     * @brief Set the actions the policy applies to.
     * @param [in] actions the actions.
     */
    void
    SetActions(std::vector<RowPolicyAction>&& actions);

    /**
     * @brief Filter expression restricting which rows are visible to the action.
     * @return the using expression.
     */
    const std::string&
    UsingExpr() const;

    /**
     * @brief Set the using expression.
     * @param [in] expr the using expression.
     */
    void
    SetUsingExpr(const std::string& expr);

    /**
     * @brief Expression restricting which rows may be written by the action.
     * @return the check expression.
     */
    const std::string&
    CheckExpr() const;

    /**
     * @brief Set the check expression.
     * @param [in] expr the check expression.
     */
    void
    SetCheckExpr(const std::string& expr);

    /**
     * @brief Description of the policy.
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
     * @brief Creation time of the policy in milliseconds; zero means unavailable.
     * @return the creation time.
     */
    int64_t
    CreatedAt() const;

    /**
     * @brief Set the creation time of the policy.
     * @param [in] created_at the creation time in milliseconds.
     */
    void
    SetCreatedAt(int64_t created_at);

 private:
    std::string policy_name_;
    RowPolicyType policy_type_{RowPolicyType::UNKNOWN};
    std::vector<RowPolicyAction> actions_;
    std::string using_expr_;
    std::string check_expr_;
    std::string description_;
    int64_t created_at_{0};
};

}  // namespace milvus
