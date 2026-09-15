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

#include <unordered_map>

#include "../../types/IDArray.h"
#include "./DQLRequestBase.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::Get()
 * @par Example
 * @code
 * milvus::GetResponse response;
 * milvus::GetRequest request;
 * request.WithCollectionName("demo").WithIDs(std::vector<int64_t>{1, 2, 3}).WithOutputFields({"id"});
 * auto status = client->Get(request, response);
 * @endcode
 */
class MILVUS_SDK_API GetRequest : public DQLRequestBase<GetRequest> {
 public:
    /**
     * @brief Constructor
     */
    GetRequest() = default;

    /**
     * @brief Get id array.
     * @return the i ds.
     */
    const IDArray&
    IDs() const;

    /**
     * @brief Set id array.
     * Note: this method will reset the id array.
     * @param [in] id_array the ID array.
     */
    void
    SetIDs(std::vector<int64_t>&& id_array);

    /**
     * @brief Set id array.
     * Note: this method will reset the id array.
     * @param [in] id_array the ID array.
     */
    void
    SetIDs(std::vector<std::string>&& id_array);

    /**
     * @brief Set id array.
     * Note: this method will reset the id array.
     * @param [in] id_array the ID array.
     */
    GetRequest&
    WithIDs(std::vector<int64_t>&& id_array);

    /**
     * @brief Set id array.
     * Note: this method will reset the id array.
     * @param [in] id_array the ID array.
     */
    GetRequest&
    WithIDs(std::vector<std::string>&& id_array);

 private:
    IDArray ids_;
};

}  // namespace milvus
