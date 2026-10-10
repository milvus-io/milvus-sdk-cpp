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

#include "milvus/request/import/CloudDescribeImportRequest.h"

namespace milvus {

const std::string&
CloudDescribeImportRequest::ClusterId() const {
    return cluster_id_;
}

void
CloudDescribeImportRequest::SetClusterId(const std::string& cluster_id) {
    cluster_id_ = cluster_id;
}

CloudDescribeImportRequest&
CloudDescribeImportRequest::WithClusterId(const std::string& cluster_id) {
    SetClusterId(cluster_id);
    return *this;
}

const std::string&
CloudDescribeImportRequest::ProjectId() const {
    return project_id_;
}

void
CloudDescribeImportRequest::SetProjectId(const std::string& project_id) {
    project_id_ = project_id;
}

CloudDescribeImportRequest&
CloudDescribeImportRequest::WithProjectId(const std::string& project_id) {
    SetProjectId(project_id);
    return *this;
}

const std::string&
CloudDescribeImportRequest::RegionId() const {
    return region_id_;
}

void
CloudDescribeImportRequest::SetRegionId(const std::string& region_id) {
    region_id_ = region_id;
}

CloudDescribeImportRequest&
CloudDescribeImportRequest::WithRegionId(const std::string& region_id) {
    SetRegionId(region_id);
    return *this;
}

const std::string&
CloudDescribeImportRequest::JobId() const {
    return job_id_;
}

void
CloudDescribeImportRequest::SetJobId(const std::string& job_id) {
    job_id_ = job_id;
}

CloudDescribeImportRequest&
CloudDescribeImportRequest::WithJobId(const std::string& job_id) {
    SetJobId(job_id);
    return *this;
}

nlohmann::json
CloudDescribeImportRequest::ToJson() const {
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
    payload["jobId"] = job_id_;
    return payload;
}

}  // namespace milvus
