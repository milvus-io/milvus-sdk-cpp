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

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>

#include "./CollectionCacheKey.h"
#include "milvus/Status.h"
#include "milvus/types/CollectionDesc.h"

namespace milvus {

class SchemaCache {
 public:
    static constexpr size_t default_capacity = 4096;
    using Loader = std::function<Status(CollectionDescPtr&)>;

    explicit SchemaCache(size_t capacity = default_capacity);

    static SchemaCache&
    GetInstance();

    // Completed schemas are shared by collection key. Concurrent loads are coalesced only when
    // load_scope is the same stable per-client identity, so one client's RPC deadline, credentials,
    // or connection state cannot control another client's schema load.
    Status
    GetOrLoad(const std::string& endpoint, const std::string& db_name, const std::string& collection_name,
              bool force_update, const void* load_scope, const Loader& loader, CollectionDescPtr& desc);

    bool
    Get(const std::string& endpoint, const std::string& db_name, const std::string& collection_name,
        CollectionDescPtr& desc);

    void
    Set(const std::string& endpoint, const std::string& db_name, const std::string& collection_name,
        CollectionDescPtr desc);

    void
    Invalidate(const std::string& endpoint, const std::string& db_name, const std::string& collection_name);

    void
    InvalidateDb(const std::string& endpoint, const std::string& db_name);

    void
    Clear();

    size_t
    Size() const;

 private:
    struct Entry {
        Entry(CollectionDescPtr desc, uint64_t last_access) : desc_(std::move(desc)), last_access_(last_access) {
        }

        CollectionDescPtr desc_;
        std::atomic<uint64_t> last_access_;
    };

    struct LoadState {
        std::mutex mutex_;
        std::condition_variable cv_;
        std::atomic<bool> invalidated_{false};
        bool completed_ = false;
        Status status_;
        CollectionDescPtr desc_;
    };

    struct LoadKey {
        CollectionCacheKey collection_key_;
        const void* scope_;

        bool
        operator==(const LoadKey& other) const {
            return collection_key_ == other.collection_key_ && scope_ == other.scope_;
        }
    };

    struct LoadKeyHash {
        size_t
        operator()(const LoadKey& key) const {
            size_t seed = CollectionCacheKeyHash{}(key.collection_key_);
            seed ^= std::hash<const void*>{}(key.scope_) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            return seed;
        }
    };

    using EntryPtr = std::shared_ptr<Entry>;
    using LoadStatePtr = std::shared_ptr<LoadState>;

    bool
    getCached(const CollectionCacheKey& key, CollectionDescPtr& desc);

    void
    setCacheNoLocked(const CollectionCacheKey& key, CollectionDescPtr desc);

    void
    invalidateLoad(const CollectionCacheKey& key);

    void
    invalidateDbLoads(const CollectionCacheKey& prefix);

    void
    invalidateAllLoads();

    uint64_t
    nextAccess();

    void
    touch(const EntryPtr& entry);

    void
    evictIfNeededLocked();

    size_t capacity_;
    mutable std::shared_timed_mutex mutex_;
    std::atomic<uint64_t> access_sequence_{0};
    std::unordered_map<CollectionCacheKey, EntryPtr, CollectionCacheKeyHash> cache_;

    std::mutex loading_mutex_;
    std::unordered_map<LoadKey, LoadStatePtr, LoadKeyHash> loading_;
};

}  // namespace milvus
