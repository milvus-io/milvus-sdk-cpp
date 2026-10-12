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

#include "milvus/request/import/VolumeImportRequest.h"

#include <utility>

namespace milvus {

const std::string&
VolumeImportRequest::ClusterId() const {
    return cluster_id_;
}

void
VolumeImportRequest::SetClusterId(const std::string& cluster_id) {
    cluster_id_ = cluster_id;
}

VolumeImportRequest&
VolumeImportRequest::WithClusterId(const std::string& cluster_id) {
    SetClusterId(cluster_id);
    return *this;
}

const std::string&
VolumeImportRequest::VolumeName() const {
    return volume_name_;
}

void
VolumeImportRequest::SetVolumeName(const std::string& volume_name) {
    volume_name_ = volume_name;
}

VolumeImportRequest&
VolumeImportRequest::WithVolumeName(const std::string& volume_name) {
    SetVolumeName(volume_name);
    return *this;
}

const std::vector<std::vector<std::string>>&
VolumeImportRequest::DataPaths() const {
    return data_paths_;
}

void
VolumeImportRequest::SetDataPaths(std::vector<std::vector<std::string>>&& data_paths) {
    data_paths_ = std::move(data_paths);
}

VolumeImportRequest&
VolumeImportRequest::WithDataPaths(std::vector<std::vector<std::string>>&& data_paths) {
    SetDataPaths(std::move(data_paths));
    return *this;
}

nlohmann::json
VolumeImportRequest::ToJson() const {
    auto payload = BaseImportRequest::ToJson();
    if (!cluster_id_.empty()) {
        payload["clusterId"] = cluster_id_;
    }
    payload["collectionName"] = collection_name_;
    if (!db_name_.empty()) {
        payload["dbName"] = db_name_;
    }
    if (!partition_name_.empty()) {
        payload["partitionName"] = partition_name_;
    }
    if (!volume_name_.empty()) {
        payload["volumeName"] = volume_name_;
    }
    payload["dataPaths"] = data_paths_;
    return payload;
}

}  // namespace milvus
