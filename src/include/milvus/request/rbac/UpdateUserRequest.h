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

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::UpdateUser().
 */
class MILVUS_SDK_API UpdateUserRequest {
 public:
    /**
     * @brief Constructor
     */
    UpdateUserRequest() = default;

    /**
     * @brief Get the user name.
     * @return the user name.
     */
    const std::string&
    UserName() const;

    /**
     * @brief Set the user name.
     *
     * @param [in] user_name
     */
    void
    SetUserName(const std::string& user_name);

    /**
     * @brief Set the user name.
     *
     * @param [in] user_name
     */
    UpdateUserRequest&
    WithUserName(const std::string& user_name);

    /**
     * @brief Get the user description.
     * @return the description.
     */
    const std::string&
    Description() const;

    /**
     * @brief Set the user description.
     *
     * @param [in] description
     */
    void
    SetDescription(const std::string& description);

    /**
     * @brief Set the user description.
     *
     * @param [in] description
     */
    UpdateUserRequest&
    WithDescription(const std::string& description);

 private:
    std::string user_name_;
    std::string description_;
};

}  // namespace milvus
