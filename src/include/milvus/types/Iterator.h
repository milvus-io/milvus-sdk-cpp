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

#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

#include "QueryResults.h"
#include "SearchResults.h"

namespace milvus {

/**
 * @brief Base class for MilvusClient::QueryIterator() and SearchIterator().
 */
template <typename T>
class Iterator {
 protected:
    Iterator() = default;

 public:
    virtual ~Iterator() = default;

    /**
     * @brief Get next batch of results.
     * Note: this method is not designed to be called in multi-thread, it is not thread-safe.
     *
     * @return QueryResults or SingleResult
     */
    virtual Status
    Next(T& results) = 0;
};

extern template class Iterator<QueryResults>;
extern template class Iterator<SingleResult>;

using QueryIterator = Iterator<QueryResults>;
using SearchIterator = Iterator<SingleResult>;

using QueryIteratorPtr = std::shared_ptr<QueryIterator>;
using SearchIteratorPtr = std::shared_ptr<SearchIterator>;

}  // namespace milvus
