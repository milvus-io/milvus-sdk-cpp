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

#include "milvus/utils/CacheConfig.h"

#include <mutex>

#include "./cache/CollectionTsCache.h"
#include "./cache/SchemaCache.h"

namespace milvus {

namespace {

std::mutex capacity_mutex;

}  // namespace

Status
CacheConfig::SetCapacity(size_t capacity) {
    if (capacity == 0) {
        return {StatusCode::INVALID_ARGUMENT, "Cache capacity must be greater than zero"};
    }
    std::lock_guard<std::mutex> lock(capacity_mutex);
    SchemaCache::GetInstance().SetCapacity(capacity);
    CollectionTsCache::GetInstance().SetCapacity(capacity);
    return Status::OK();
}

size_t
CacheConfig::GetCapacity() {
    // Reads only SchemaCache's capacity, which is internally protected by its
    // own lock and returned as a single size_t; no outer lock is needed.
    return SchemaCache::GetInstance().Capacity();
}

}  // namespace milvus
