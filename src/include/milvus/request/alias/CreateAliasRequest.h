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

#include "./AliasRequestBase.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::CreateAlias().
 */
class CreateAliasRequest : public AliasRequestBase<CreateAliasRequest> {
 public:
    /**
     * @brief Constructor
     */
    CreateAliasRequest() = default;

    /**
     * @brief Set name of the alias.
     */
    const std::string&
    Alias() const;

    /**
     * @brief Set name of the alias.
     */
    void
    SetAlias(const std::string& alias);

    /**
     * @brief Set name of the alias.
     */
    CreateAliasRequest&
    WithAlias(const std::string& alias);

 protected:
    std::string alias_;
};

}  // namespace milvus
