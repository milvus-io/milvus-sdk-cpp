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

#include "milvus/response/utility/GetServerVersionResponse.h"

namespace milvus {

const std::string&
GetServerVersionResponse::Version() const {
    return version_;
}

void
GetServerVersionResponse::SetVersion(const std::string& version) {
    version_ = version;
}

const std::string&
GetServerVersionResponse::BuildTime() const {
    return build_time_;
}

void
GetServerVersionResponse::SetBuildTime(const std::string& build_time) {
    build_time_ = build_time;
}

const std::string&
GetServerVersionResponse::GitCommit() const {
    return git_commit_;
}

void
GetServerVersionResponse::SetGitCommit(const std::string& git_commit) {
    git_commit_ = git_commit;
}

const std::string&
GetServerVersionResponse::GoVersion() const {
    return go_version_;
}

void
GetServerVersionResponse::SetGoVersion(const std::string& go_version) {
    go_version_ = go_version;
}

const std::string&
GetServerVersionResponse::DeployMode() const {
    return deploy_mode_;
}

void
GetServerVersionResponse::SetDeployMode(const std::string& deploy_mode) {
    deploy_mode_ = deploy_mode;
}

}  // namespace milvus
