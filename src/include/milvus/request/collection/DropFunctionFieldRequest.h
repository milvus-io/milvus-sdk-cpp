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

#include "./CollectionRequestBase.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::DropFunctionField()
 */
class MILVUS_SDK_API DropFunctionFieldRequest : public CollectionRequestBase<DropFunctionFieldRequest> {
 public:
    /**
     * @brief Constructor
     */
    DropFunctionFieldRequest() = default;

    /**
     * @brief Name of the function to drop together with its output fields.
     */
    const std::string&
    FunctionName() const;

    /**
     * @brief Set the name of the function to drop together with its output fields.
     */
    void
    SetFunctionName(std::string function_name);

    /**
     * @brief Set the name of the function to drop together with its output fields.
     */
    DropFunctionFieldRequest&
    WithFunctionName(std::string function_name);

 private:
    std::string function_name_;
};

}  // namespace milvus
