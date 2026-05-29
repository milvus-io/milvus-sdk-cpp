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
 * @brief Used by MilvusClientV2::FlushAll()
 */
class MILVUS_SDK_API FlushAllRequest {
 public:
    /**
     * @brief Constructor
     */
    FlushAllRequest() = default;

    /**
     * @brief Database name in which collections are flushed.
     */
    const std::string&
    DatabaseName() const;

    /**
     * @brief Set database name in which collections are flushed.
     */
    void
    SetDatabaseName(const std::string& db_name);

    /**
     * @brief Set database name in which collections are flushed.
     */
    FlushAllRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Get milliseconds to wait the flush-all action done.
     */
    int64_t
    WaitFlushedMs() const;

    /**
     * @brief Set milliseconds to wait the flush-all action done. Default value is 0, which means forever.
     */
    void
    SetWaitFlushedMs(int64_t ms);

    /**
     * @brief Set milliseconds to wait the flush-all action done. Default value is 0, which means forever.
     */
    FlushAllRequest&
    WithWaitFlushedMs(int64_t ms);

 private:
    std::string db_name_;
    int64_t wait_flushed_ms_{0};
};

}  // namespace milvus
