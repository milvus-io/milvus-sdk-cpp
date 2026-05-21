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

#include "CollectionRequestBase.h"
#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::RefreshLoad()
 */
class MILVUS_SDK_API RefreshLoadRequest : public CollectionRequestBase<RefreshLoadRequest> {
 public:
    /**
     * @brief Constructor
     */
    RefreshLoadRequest() = default;

    /**
     * @brief Sync mode.
     */
    bool
    Sync() const;

    /**
     * @brief Set sync mode. Default value is true.
     * True: wait the collection refresh to complete.
     * False: return immediately.
     */
    void
    SetSync(bool sync);

    /**
     * @brief Set sync mode. Default value is true.
     * True: wait the collection refresh to complete.
     * False: return immediately.
     */
    RefreshLoadRequest&
    WithSync(bool sync);

    /**
     * @brief Timeout in milliseconds.
     */
    int64_t
    TimeoutMs() const;

    /**
     * @brief Set timeout in milliseconds. Default value is 60000ms. Only works when Sync() is true.
     */
    void
    SetTimeoutMs(int64_t timeout_ms);

    /**
     * @brief Set timeout in milliseconds. Default value is 60000ms. Only works when Sync() is true.
     */
    RefreshLoadRequest&
    WithTimeoutMs(int64_t timeout_ms);

 private:
    bool sync_{true};
    int64_t timeout_ms_{60000};
};

}  // namespace milvus
