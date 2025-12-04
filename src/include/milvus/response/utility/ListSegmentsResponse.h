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

#include "milvus/types/SegmentInfo.h"

namespace milvus {

/**
 * @brief Used by MilvusClientV2::ListPersistentSegments() and ListQuerySegments()
 */
template <typename T>
class ListSegmentsResponse {
 public:
    /**
     * @brief Constructor
     */
    ListSegmentsResponse() = default;

    // Getter and Setter for results_
    const T&
    Result() const;

    void
    SetResult(T&& results);

 private:
    T segments_info_;
};

extern template class ListSegmentsResponse<SegmentsInfo>;
extern template class ListSegmentsResponse<QuerySegmentsInfo>;

using ListPersistentSegmentsResponse = ListSegmentsResponse<SegmentsInfo>;
using ListQuerySegmentsResponse = ListSegmentsResponse<QuerySegmentsInfo>;

}  // namespace milvus
