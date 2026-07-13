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

#include "MilvusClientV2SessionImpl.h"

#include <utility>

#include "MilvusClientV2Impl.h"

namespace milvus {

MilvusClientV2SessionImpl::MilvusClientV2SessionImpl(std::shared_ptr<MilvusClientV2Impl> parent, std::string cluster_id)
    : parent_(std::move(parent)), cluster_id_(std::move(cluster_id)) {
}

const std::string&
MilvusClientV2SessionImpl::ClusterID() const {
    return cluster_id_;
}

Status
MilvusClientV2SessionImpl::Search(const SearchRequest& request, SearchResponse& response) {
    std::shared_ptr<MilvusClientV2Impl> parent;
    auto status = getParent(parent);
    return status.IsOk() ? parent->search(request, response, cluster_id_) : status;
}

Status
MilvusClientV2SessionImpl::SearchIterator(const SearchIteratorRequest& request, SearchIteratorPtr& iterator) {
    std::shared_ptr<MilvusClientV2Impl> parent;
    auto status = getParent(parent);
    if (!status.IsOk()) {
        return status;
    }
    auto request_copy = request;
    return parent->searchIterator(request_copy, iterator, cluster_id_);
}

Status
MilvusClientV2SessionImpl::HybridSearch(const HybridSearchRequest& request, HybridSearchResponse& response) {
    std::shared_ptr<MilvusClientV2Impl> parent;
    auto status = getParent(parent);
    return status.IsOk() ? parent->hybridSearch(request, response, cluster_id_) : status;
}

Status
MilvusClientV2SessionImpl::Query(const QueryRequest& request, QueryResponse& response) {
    std::shared_ptr<MilvusClientV2Impl> parent;
    auto status = getParent(parent);
    return status.IsOk() ? parent->query(request, response, cluster_id_) : status;
}

Status
MilvusClientV2SessionImpl::Get(const GetRequest& request, GetResponse& response) {
    std::shared_ptr<MilvusClientV2Impl> parent;
    auto status = getParent(parent);
    return status.IsOk() ? parent->get(request, response, cluster_id_) : status;
}

Status
MilvusClientV2SessionImpl::QueryIterator(const QueryIteratorRequest& request, QueryIteratorPtr& iterator) {
    std::shared_ptr<MilvusClientV2Impl> parent;
    auto status = getParent(parent);
    if (!status.IsOk()) {
        return status;
    }
    auto request_copy = request;
    return parent->queryIterator(request_copy, iterator, cluster_id_);
}

void
MilvusClientV2SessionImpl::Close() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    parent_.reset();
}

Status
MilvusClientV2SessionImpl::getParent(std::shared_ptr<MilvusClientV2Impl>& parent) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
        return {StatusCode::NOT_CONNECTED, "MilvusClient session is closed"};
    }
    parent = parent_;
    return Status::OK();
}

}  // namespace milvus
