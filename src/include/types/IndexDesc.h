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
#include <unordered_map>

namespace milvus {

/**
 * @brief Index description. Used by CreateIndex() and DescribeIndex().
 */
class IndexDesc {
 public:
 private:
    /**
     * @brief Name of the field.
     */
    std::string field_name_;

    /**
     * @brief Internal name of the index. Reserved for funture feature: multiple indice in one field.
     * Only avaiable for DescribeIndex(). No need to specify it for CreateIndex().
     */
    std::string index_name_;

    /**
     * @brief Internal id of the index. Reserved for funture feature: multiple indice in one field.
     * Only avaiable for DescribeIndex(). No need to specify it for CreateIndex().
     */
    int64_t index_id_;

    /**
     * @brief Extra parameters of the index.
     */
    std::unordered_map<std::string, std::string> params_;
};

}  // namespace milvus
