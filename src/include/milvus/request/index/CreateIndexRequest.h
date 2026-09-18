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

#include "../../types/IndexDesc.h"
#include "../../types/IndexParam.h"
#include "./IndexRequestBase.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::CreateIndex()
 * @par Example
 * @code
 * milvus::IndexParam index("vector", "vector_idx", milvus::IndexType::HNSW, milvus::MetricType::L2);
 * index.AddExtraParam("M", "16");
 * auto status = client->CreateIndex(milvus::CreateIndexRequest()
 *                                       .WithCollectionName("demo")
 *                                       .WithIndexParams({std::move(index)})
 *                                       .WithSync(true));
 * @endcode
 */
class MILVUS_SDK_API CreateIndexRequest : public IndexRequestBase<CreateIndexRequest> {
 public:
    /**
     * @brief Constructor
     */
    CreateIndexRequest() = default;

    /**
     * @brief Get index params.
     * @return the index params.
     */
    const std::vector<IndexParam>&
    IndexParams() const;

    /**
     * @brief Set index params to be created.
     * @param [in] index_params the index params.
     */
    void
    SetIndexParams(std::vector<IndexParam>&& index_params);

    /**
     * @brief Set index params to be created.
     * @param [in] index_params the index params.
     */
    CreateIndexRequest&
    WithIndexParams(std::vector<IndexParam>&& index_params);

    /**
     * @brief Add an index param to be created.
     * @param [in] index_param the index param.
     */
    CreateIndexRequest&
    AddIndexParam(IndexParam&& index_param);

    /**
     * @brief Get indexes.
     * @return the indexes.
     * @deprecated Use IndexParams() instead.
     */
    [[deprecated("use IndexParams() instead")]] const std::vector<IndexDesc>&
    Indexes() const;

    /**
     * @brief Set indexes to be created.
     * @param [in] indexes the indexes.
     * @deprecated Use WithIndexParams() instead.
     */
    [[deprecated("use WithIndexParams() instead")]] void
    SetIndexes(std::vector<IndexDesc>&& indexes);

    /**
     * @brief Set indexes to be created.
     * @param [in] indexes the indexes.
     * @deprecated Use WithIndexParams() instead.
     */
    [[deprecated("use WithIndexParams() instead")]] CreateIndexRequest&
    WithIndexes(std::vector<IndexDesc>&& indexes);

    /**
     * @brief Add an index to be created.
     * @param [in] index the index.
     * @deprecated Use AddIndexParam() instead.
     */
    [[deprecated("use AddIndexParam() instead")]] CreateIndexRequest&
    AddIndex(IndexDesc&& index);

    /**
     * @brief Get sync mode.
     * True: wait the indexes are ready.
     * False: return immediately no matter the indexes are ready or not.
     * @return the sync.
     */
    bool
    Sync() const;

    /**
     * @brief Set sync mode. Default value is true.
     * True: wait the indexes are ready.
     * False: return immediately no matter the indexes are ready or not.
     * @param [in] sync the sync.
     */
    void
    SetSync(bool sync);

    /**
     * @brief Set sync mode. Default value is true.
     * True: wait the indexes are ready.
     * False: return immediately no matter the indexes are ready or not.
     * @param [in] sync the sync.
     */
    CreateIndexRequest&
    WithSync(bool sync);

    /**
     * @brief Timeout in milliseconds.
     * @return the timeout ms.
     */
    int64_t
    TimeoutMs() const;

    /**
     * @brief Set timeout in milliseconds. Default value is 60000ms. Only work when Sync() is true.
     * If the TimeoutMs is zero, the CreateIndex() will call DescribeIndex() to loading state,
     * until the index is fully built.
     * If the TimeoutMs is larger than zero, the CreateIndex() will break the loop after a certain of time span
     * and return a status saying the process is timeout.
     * @param [in] timeout_ms the timeout ms.
     */
    void
    SetTimeoutMs(int64_t timeout_ms);

    /**
     * @brief Set timeout in milliseconds. Default value is 60000ms. Only work when Sync() is true.
     * If the TimeoutMs is zero, the CreateIndex() will call DescribeIndex() to index state,
     * until the index is fully built.
     * If the TimeoutMs is larger than zero, the CreateIndex() will break the loop after a certain of time span
     * and return a status saying the process is timeout.
     * @param [in] timeout_ms the timeout ms.
     */
    CreateIndexRequest&
    WithTimeoutMs(int64_t timeout_ms);

 private:
    std::vector<IndexParam> index_params_;
    mutable std::vector<IndexDesc> indexes_cache_;
    bool sync_{true};
    int64_t timeout_ms_{60000};
};

}  // namespace milvus
