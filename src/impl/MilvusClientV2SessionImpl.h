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
#include <mutex>
#include <string>

#include "milvus/MilvusClientV2Session.h"

namespace milvus {

class MilvusClientV2Impl;

class MilvusClientV2SessionImpl final : public MilvusClientV2Session {
 public:
    MilvusClientV2SessionImpl(std::shared_ptr<MilvusClientV2Impl> parent, std::string cluster_id);

    const std::string&
    ClusterID() const final;

    Status
    Search(const SearchRequest& request, SearchResponse& response) final;

    Status
    SearchIterator(const SearchIteratorRequest& request, SearchIteratorPtr& iterator) final;

    Status
    HybridSearch(const HybridSearchRequest& request, HybridSearchResponse& response) final;

    Status
    Query(const QueryRequest& request, QueryResponse& response) final;

    Status
    Get(const GetRequest& request, GetResponse& response) final;

    Status
    QueryIterator(const QueryIteratorRequest& request, QueryIteratorPtr& iterator) final;

    void
    Close() noexcept final;

 private:
    Status
    getParent(std::shared_ptr<MilvusClientV2Impl>& parent) const;

 private:
    mutable std::mutex mutex_;
    std::shared_ptr<MilvusClientV2Impl> parent_;
    const std::string cluster_id_;
    bool closed_{false};
};

}  // namespace milvus
