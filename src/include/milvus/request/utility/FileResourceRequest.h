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
 * @brief Used by MilvusClientV2::AddFileResource() and RemoveFileResource()
 */
class MILVUS_SDK_API FileResourceRequest {
 public:
    FileResourceRequest() = default;

    const std::string&
    Name() const;

    void
    SetName(const std::string& name);

    FileResourceRequest&
    WithName(const std::string& name);

 private:
    std::string name_;
};

/**
 * @brief Used by MilvusClientV2::AddFileResource()
 */
class MILVUS_SDK_API AddFileResourceRequest : public FileResourceRequest {
 public:
    AddFileResourceRequest() = default;

    AddFileResourceRequest&
    WithName(const std::string& name);

    const std::string&
    Path() const;

    void
    SetPath(const std::string& path);

    AddFileResourceRequest&
    WithPath(const std::string& path);

 private:
    std::string path_;
};

using RemoveFileResourceRequest = FileResourceRequest;

/**
 * @brief Used by MilvusClientV2::ListFileResources()
 */
class MILVUS_SDK_API ListFileResourcesRequest {
 public:
    ListFileResourcesRequest() = default;
};

}  // namespace milvus
