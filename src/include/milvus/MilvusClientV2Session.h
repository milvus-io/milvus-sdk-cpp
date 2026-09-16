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

#include <memory>

#include "Status.h"
#include "milvus/Export.h"
#include "request/dql/GetRequest.h"
#include "request/dql/HybridSearchRequest.h"
#include "request/dql/QueryIteratorRequest.h"
#include "request/dql/QueryRequest.h"
#include "request/dql/SearchIteratorRequest.h"
#include "request/dql/SearchRequest.h"
#include "response/dql/QueryResponse.h"
#include "response/dql/SearchResponse.h"
#include "types/Iterator.h"

namespace milvus {

/**
 * @brief A cluster-scoped view of MilvusClientV2 exposing DQL interfaces only.
 * @par Example
 * @code
 * milvus::MilvusClientV2SessionPtr session;
 * if (client->Session("cluster-a", session).IsOk()) {
 *     milvus::SearchResponse response;
 *     milvus::SearchRequest request;
 *     request.WithCollectionName("demo").WithLimit(3);
 *     request.AddFloatVector(std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f});
 *     session->Search(request, response);
 * }
 * @endcode
 */
class MILVUS_SDK_API MilvusClientV2Session {
 public:
    virtual ~MilvusClientV2Session() = default;

    /**
     * @brief Get the target cluster identifier.
     * @return the cluster ID.
     */
    virtual const std::string&
    ClusterID() const = 0;

    /**
     * @brief Perform a vector similarity search on the target cluster.
     *
     * @param [in] request search request parameters.
     * @param [out] response search results returned by the server.
     * @return Status operation result.
     */
    virtual Status
    Search(const SearchRequest& request, SearchResponse& response) = 0;

    /**
     * @brief Create a search iterator to fetch results in batches from the target cluster.
     *
     * @param [in] request search iterator request parameters.
     * @param [out] iterator search iterator for fetching results incrementally.
     * @return Status operation result.
     */
    virtual Status
    SearchIterator(const SearchIteratorRequest& request, SearchIteratorPtr& iterator) = 0;

    /**
     * @brief Perform a hybrid (multi-vector, reranked) search on the target cluster.
     *
     * @param [in] request hybrid search request parameters.
     * @param [out] response hybrid search results returned by the server.
     * @return Status operation result.
     */
    virtual Status
    HybridSearch(const HybridSearchRequest& request, HybridSearchResponse& response) = 0;

    /**
     * @brief Query entities on the target cluster with an expression filter.
     *
     * @param [in] request query request parameters.
     * @param [out] response query results returned by the server.
     * @return Status operation result.
     */
    virtual Status
    Query(const QueryRequest& request, QueryResponse& response) = 0;

    /**
     * @brief Query entities by primary key on the target cluster.
     *
     * @param [in] request get request parameters.
     * @param [out] response get results returned by the server.
     * @return Status operation result.
     */
    virtual Status
    Get(const GetRequest& request, GetResponse& response) = 0;

    /**
     * @brief Create a query iterator to fetch query results in batches from the target cluster.
     *
     * @param [in] request query iterator request parameters.
     * @param [out] iterator query iterator for fetching results incrementally.
     * @return Status operation result.
     */
    virtual Status
    QueryIterator(const QueryIteratorRequest& request, QueryIteratorPtr& iterator) = 0;

    /**
     * @brief Close this session view without disconnecting the parent client.
     */
    virtual void
    Close() noexcept = 0;
};

using MilvusClientV2SessionPtr = std::shared_ptr<MilvusClientV2Session>;

}  // namespace milvus
