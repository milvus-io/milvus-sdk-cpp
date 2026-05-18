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

#include <cstdint>
#include <string>

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::Optimize()
 */
class MILVUS_SDK_API OptimizeRequest {
 public:
    /**
     * @brief Constructor
     */
    OptimizeRequest() = default;

    /**
     * @brief Database name in which the collection is created.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name in which the collection is created.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set database name in which the collection is created.
     */
    OptimizeRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Name of the collection to be optimized.
     */
    const std::string&
    CollectionName() const;

    /**
     * @brief Set name of the collection to be optimized.
     */
    void
    SetCollectionName(const std::string& collection_name);

    /**
     * @brief Set name of the collection to be optimized.
     */
    OptimizeRequest&
    WithCollectionName(const std::string& collection_name);

    /**
     * @brief Target segment size, such as "512MB" or "1GB".
     */
    const std::string&
    TargetSize() const;

    /**
     * @brief Set target segment size.
     */
    void
    SetTargetSize(const std::string& target_size);

    /**
     * @brief Set target segment size.
     */
    OptimizeRequest&
    WithTargetSize(const std::string& target_size);

    /**
     * @brief Run optimization asynchronously.
     */
    bool
    Async() const;

    /**
     * @brief Set async option.
     */
    void
    SetAsync(bool async);

    /**
     * @brief Set async option.
     */
    OptimizeRequest&
    WithAsync(bool async);

    /**
     * @brief Overall task timeout in milliseconds. Zero means no overall timeout.
     */
    int64_t
    TimeoutMs() const;

    /**
     * @brief Set overall task timeout in milliseconds. Zero means no overall timeout.
     */
    void
    SetTimeoutMs(int64_t timeout_ms);

    /**
     * @brief Set overall task timeout in milliseconds. Zero means no overall timeout.
     */
    OptimizeRequest&
    WithTimeoutMs(int64_t timeout_ms);

 private:
    std::string db_name_;
    std::string collection_name_;
    std::string target_size_;
    bool async_{false};
    int64_t timeout_ms_{0};
};

}  // namespace milvus
