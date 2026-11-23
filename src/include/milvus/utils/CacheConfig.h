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

#include <cstddef>

#include "milvus/Export.h"
#include "milvus/Status.h"

namespace milvus {

/**
 * @brief Default LRU entry bound shared by the schema and collection-timestamp caches.
 */
constexpr size_t DEFAULT_CAPACITY = 4096;

/**
 * @brief Static entry points to tune the in-process SDK caches.
 *
 * The schema and collection-timestamp caches are process-wide singletons that
 * keep a bounded LRU of per-collection entries. Both default to 4096 entries
 * and evict the least recently used entry once the configured capacity is
 * exceeded. Call SetCapacity before heavy workloads to raise or lower the
 * bound; shrinking below the current size evicts entries immediately.
 *
 * Note: the collection-timestamp cache backs the SESSION consistency level.
 * Once a client tracks more than the configured number of collections, the
 * least recently used timestamps are evicted and SESSION reads for those
 * collections fall back to eventual consistency until a later mutation stores
 * a fresh timestamp. This mirrors pymilvus, which bounds both caches at 4096.
 *
 * The two caches are sized by one shared capacity on purpose (matching
 * pymilvus' single-knob default), even though a schema entry is far heavier
 * than a timestamp entry. Growing the capacity raises SESSION coverage and
 * schema retention together; callers needing very large SESSION working sets
 * should weigh the additional schema memory against their collection count.
 */
class MILVUS_SDK_API CacheConfig {
 public:
    /**
     * @brief Set the maximum number of cached collection schemas and timestamps.
     * @param [in] capacity the new capacity for both caches; must be greater than zero.
     * @return a Status indicating success or an invalid argument.
     */
    static Status
    SetCapacity(size_t capacity);

    /**
     * @brief Get the current maximum number of cached collection schemas and timestamps.
     *
     * Returns the schema cache capacity. Both caches share the same value when
     * configured through SetCapacity; callers who resized a cache directly
     * should read that cache's own Capacity() instead.
     * @return the current schema cache capacity.
     */
    static size_t
    GetCapacity();
};

}  // namespace milvus
