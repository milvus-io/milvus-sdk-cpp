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
 * @brief Used by MilvusClientV2::GetFlushAllState()
 */
class MILVUS_SDK_API GetFlushAllStateRequest {
 public:
    /**
     * @brief Constructor
     */
    GetFlushAllStateRequest() = default;

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
    GetFlushAllStateRequest&
    WithDatabaseName(const std::string& db_name);

    /**
     * @brief Get flush-all timestamp returned by FlushAll().
     */
    uint64_t
    FlushAllTs() const;

    /**
     * @brief Set flush-all timestamp returned by FlushAll().
     */
    void
    SetFlushAllTs(uint64_t flush_all_ts);

    /**
     * @brief Set flush-all timestamp returned by FlushAll().
     */
    GetFlushAllStateRequest&
    WithFlushAllTs(uint64_t flush_all_ts);

 private:
    std::string db_name_;
    uint64_t flush_all_ts_{0};
};

}  // namespace milvus
