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

#include "milvus/request/import/CloudListImportJobsRequest.h"

namespace milvus {

const std::string&
CloudListImportJobsRequest::ClusterId() const {
    return cluster_id_;
}

void
CloudListImportJobsRequest::SetClusterId(const std::string& cluster_id) {
    cluster_id_ = cluster_id;
}

CloudListImportJobsRequest&
CloudListImportJobsRequest::WithClusterId(const std::string& cluster_id) {
    SetClusterId(cluster_id);
    return *this;
}

const std::string&
CloudListImportJobsRequest::ProjectId() const {
    return project_id_;
}

void
CloudListImportJobsRequest::SetProjectId(const std::string& project_id) {
    project_id_ = project_id;
}

CloudListImportJobsRequest&
CloudListImportJobsRequest::WithProjectId(const std::string& project_id) {
    SetProjectId(project_id);
    return *this;
}

const std::string&
CloudListImportJobsRequest::RegionId() const {
    return region_id_;
}

void
CloudListImportJobsRequest::SetRegionId(const std::string& region_id) {
    region_id_ = region_id;
}

CloudListImportJobsRequest&
CloudListImportJobsRequest::WithRegionId(const std::string& region_id) {
    SetRegionId(region_id);
    return *this;
}

int64_t
CloudListImportJobsRequest::PageSize() const {
    return page_size_;
}

void
CloudListImportJobsRequest::SetPageSize(int64_t page_size) {
    page_size_ = page_size;
}

CloudListImportJobsRequest&
CloudListImportJobsRequest::WithPageSize(int64_t page_size) {
    SetPageSize(page_size);
    return *this;
}

int64_t
CloudListImportJobsRequest::CurrentPage() const {
    return current_page_;
}

void
CloudListImportJobsRequest::SetCurrentPage(int64_t current_page) {
    current_page_ = current_page;
}

CloudListImportJobsRequest&
CloudListImportJobsRequest::WithCurrentPage(int64_t current_page) {
    SetCurrentPage(current_page);
    return *this;
}

nlohmann::json
CloudListImportJobsRequest::ToJson() const {
    nlohmann::json payload = nlohmann::json::object();
    if (!cluster_id_.empty()) {
        payload["clusterId"] = cluster_id_;
    }
    if (!project_id_.empty()) {
        payload["projectId"] = project_id_;
    }
    if (!region_id_.empty()) {
        payload["regionId"] = region_id_;
    }
    if (page_size_ != 0) {
        payload["pageSize"] = page_size_;
    }
    if (current_page_ != 0) {
        payload["currentPage"] = current_page_;
    }
    return payload;
}

}  // namespace milvus
