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

namespace milvus {

/**
 * @brief Used by MilvusClientV2::ListRLSPrincipals()
 */
class MILVUS_SDK_API ListRLSPrincipalsResponse {
 public:
    /**
     * @brief Constructor
     */
    ListRLSPrincipalsResponse() = default;

    /**
     * @brief Get names of principals that have stored tags in the collection.
     * @return the principal names.
     */
    const std::vector<std::string>&
    PrincipalNames() const;

    /**
     * @brief Set names of principals that have stored tags in the collection.
     * @param [in] principal_names the principal names.
     */
    void
    SetPrincipalNames(std::vector<std::string>&& principal_names);

 private:
    std::vector<std::string> principal_names_;
};

}  // namespace milvus
