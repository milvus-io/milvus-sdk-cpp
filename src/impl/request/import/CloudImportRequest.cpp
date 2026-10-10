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

#include "milvus/request/import/CloudImportRequest.h"

#include <utility>

namespace milvus {

const std::string&
CloudImportRequest::ClusterId() const {
    return cluster_id_;
}

void
CloudImportRequest::SetClusterId(const std::string& cluster_id) {
    cluster_id_ = cluster_id;
}

CloudImportRequest&
CloudImportRequest::WithClusterId(const std::string& cluster_id) {
    SetClusterId(cluster_id);
    return *this;
}

const std::string&
CloudImportRequest::ProjectId() const {
    return project_id_;
}

void
CloudImportRequest::SetProjectId(const std::string& project_id) {
    project_id_ = project_id;
}

CloudImportRequest&
CloudImportRequest::WithProjectId(const std::string& project_id) {
    SetProjectId(project_id);
    return *this;
}

const std::string&
CloudImportRequest::RegionId() const {
    return region_id_;
}

void
CloudImportRequest::SetRegionId(const std::string& region_id) {
    region_id_ = region_id;
}

CloudImportRequest&
CloudImportRequest::WithRegionId(const std::string& region_id) {
    SetRegionId(region_id);
    return *this;
}

const std::vector<std::vector<std::string>>&
CloudImportRequest::ObjectUrls() const {
    return object_urls_;
}

void
CloudImportRequest::SetObjectUrls(std::vector<std::vector<std::string>>&& object_urls) {
    object_urls_ = std::move(object_urls);
}

CloudImportRequest&
CloudImportRequest::WithObjectUrls(std::vector<std::vector<std::string>>&& object_urls) {
    SetObjectUrls(std::move(object_urls));
    return *this;
}

const std::string&
CloudImportRequest::ObjectUrl() const {
    return object_url_;
}

void
CloudImportRequest::SetObjectUrl(const std::string& object_url) {
    object_url_ = object_url;
}

CloudImportRequest&
CloudImportRequest::WithObjectUrl(const std::string& object_url) {
    SetObjectUrl(object_url);
    return *this;
}

const std::string&
CloudImportRequest::AccessKey() const {
    return access_key_;
}

void
CloudImportRequest::SetAccessKey(const std::string& access_key) {
    access_key_ = access_key;
}

CloudImportRequest&
CloudImportRequest::WithAccessKey(const std::string& access_key) {
    SetAccessKey(access_key);
    return *this;
}

const std::string&
CloudImportRequest::SecretKey() const {
    return secret_key_;
}

void
CloudImportRequest::SetSecretKey(const std::string& secret_key) {
    secret_key_ = secret_key;
}

CloudImportRequest&
CloudImportRequest::WithSecretKey(const std::string& secret_key) {
    SetSecretKey(secret_key);
    return *this;
}

const std::string&
CloudImportRequest::Token() const {
    return token_;
}

void
CloudImportRequest::SetToken(const std::string& token) {
    token_ = token;
}

CloudImportRequest&
CloudImportRequest::WithToken(const std::string& token) {
    SetToken(token);
    return *this;
}

nlohmann::json
CloudImportRequest::ToJson() const {
    auto payload = BaseImportRequest::ToJson();
    if (!cluster_id_.empty()) {
        payload["clusterId"] = cluster_id_;
    }
    if (!project_id_.empty()) {
        payload["projectId"] = project_id_;
    }
    if (!region_id_.empty()) {
        payload["regionId"] = region_id_;
    }
    payload["collectionName"] = collection_name_;
    if (!db_name_.empty()) {
        payload["dbName"] = db_name_;
    }
    if (!partition_name_.empty()) {
        payload["partitionName"] = partition_name_;
    }
    if (!object_urls_.empty()) {
        payload["objectUrls"] = object_urls_;
    }
    if (!object_url_.empty()) {
        payload["objectUrl"] = object_url_;
    }
    if (!access_key_.empty()) {
        payload["accessKey"] = access_key_;
    }
    if (!secret_key_.empty()) {
        payload["secretKey"] = secret_key_;
    }
    if (!token_.empty()) {
        payload["token"] = token_;
    }
    return payload;
}

}  // namespace milvus
