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

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::GetServerVersionV2()
 */
class MILVUS_SDK_API GetServerVersionResponse {
 public:
    /**
     * @brief Constructor
     */
    GetServerVersionResponse() = default;

    /**
     * @brief Version of the Milvus server.
     */
    const std::string&
    Version() const;

    /**
     * @brief Set version of the Milvus server.
     */
    void
    SetVersion(const std::string& version);

    /**
     * @brief Build time of the Milvus server.
     */
    const std::string&
    BuildTime() const;

    /**
     * @brief Set build time of the Milvus server.
     */
    void
    SetBuildTime(const std::string& build_time);

    /**
     * @brief Git commit of the Milvus server build.
     */
    const std::string&
    GitCommit() const;

    /**
     * @brief Set git commit of the Milvus server build.
     */
    void
    SetGitCommit(const std::string& git_commit);

    /**
     * @brief Go version used to build the Milvus server.
     */
    const std::string&
    GoVersion() const;

    /**
     * @brief Set Go version used to build the Milvus server.
     */
    void
    SetGoVersion(const std::string& go_version);

    /**
     * @brief Deploy mode of the Milvus server.
     */
    const std::string&
    DeployMode() const;

    /**
     * @brief Set deploy mode of the Milvus server.
     */
    void
    SetDeployMode(const std::string& deploy_mode);

 private:
    std::string version_;
    std::string build_time_;
    std::string git_commit_;
    std::string go_version_;
    std::string deploy_mode_;
};

}  // namespace milvus
