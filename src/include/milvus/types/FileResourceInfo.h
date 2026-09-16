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
 * @brief File resource information.
 */
class MILVUS_SDK_API FileResourceInfo {
    /**
     * @brief Get the resource name.
     * @return the name.
     */
 public:
    const std::string&
    Name() const;

    /**
     * @brief Set the resource name.
     *
     * @param [in] name
     */
    void
    SetName(std::string name);

    /**
     * @brief Get the file path of the resource.
     * @return the path.
     */
    const std::string&
    Path() const;

    /**
     * @brief Set the file path of the resource.
     *
     * @param [in] path
     */
    void
    SetPath(std::string path);

 private:
    std::string name_;
    std::string path_;
};

}  // namespace milvus
