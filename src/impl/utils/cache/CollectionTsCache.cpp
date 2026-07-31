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

#include "./CollectionTsCache.h"

#include <algorithm>

namespace milvus {

CollectionTsCache&
CollectionTsCache::GetInstance() {
    static CollectionTsCache instance;
    return instance;
}

uint64_t
CollectionTsCache::Get(const std::string& endpoint, const std::string& db_name, const std::string& collection_name) {
    const auto key = CollectionCacheKey::Create(endpoint, db_name, collection_name);
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        return 0;
    }
    return it->second;
}

void
CollectionTsCache::Set(const std::string& endpoint, const std::string& db_name, const std::string& collection_name,
                       uint64_t ts) {
    if (ts == 0) {
        return;
    }

    const auto key = CollectionCacheKey::Create(endpoint, db_name, collection_name);
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        if (ts > it->second) {
            it->second = ts;
        }
        return;
    }

    cache_.emplace(key, ts);
}

void
CollectionTsCache::Invalidate(const std::string& endpoint, const std::string& db_name,
                              const std::string& collection_name) {
    const auto key = CollectionCacheKey::Create(endpoint, db_name, collection_name);
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    cache_.erase(key);
}

void
CollectionTsCache::InvalidateDb(const std::string& endpoint, const std::string& db_name) {
    const auto prefix = CollectionCacheKey::Create(endpoint, db_name, "");
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    for (auto it = cache_.begin(); it != cache_.end();) {
        if (it->first.endpoint_ == prefix.endpoint_ && it->first.db_name_ == prefix.db_name_) {
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
}

void
CollectionTsCache::Move(const std::string& endpoint, const std::string& old_db_name,
                        const std::string& old_collection_name, const std::string& new_db_name,
                        const std::string& new_collection_name) {
    transfer(CollectionCacheKey::Create(endpoint, old_db_name, old_collection_name),
             CollectionCacheKey::Create(endpoint, new_db_name, new_collection_name), true);
}

void
CollectionTsCache::Copy(const std::string& endpoint, const std::string& source_db_name,
                        const std::string& source_collection_name, const std::string& target_db_name,
                        const std::string& target_collection_name) {
    transfer(CollectionCacheKey::Create(endpoint, source_db_name, source_collection_name),
             CollectionCacheKey::Create(endpoint, target_db_name, target_collection_name), false);
}

void
CollectionTsCache::transfer(const CollectionCacheKey& source_key, const CollectionCacheKey& target_key,
                            bool drop_source) {
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);

    if (source_key == target_key) {
        return;
    }

    uint64_t latest_ts = 0;
    auto source_it = cache_.find(source_key);
    if (source_it != cache_.end()) {
        latest_ts = source_it->second;
    }
    auto target_it = cache_.find(target_key);
    if (target_it != cache_.end()) {
        latest_ts = std::max(latest_ts, target_it->second);
    }

    if (drop_source) {
        cache_.erase(source_key);
    }
    cache_.erase(target_key);
    if (latest_ts != 0) {
        cache_.emplace(target_key, latest_ts);
    }
}

void
CollectionTsCache::Clear() {
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    cache_.clear();
}

size_t
CollectionTsCache::Size() const {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    return cache_.size();
}

}  // namespace milvus
