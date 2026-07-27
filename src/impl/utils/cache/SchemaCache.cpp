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

#include "./SchemaCache.h"

#include <algorithm>
#include <exception>

namespace milvus {

SchemaCache::SchemaCache(size_t capacity) : capacity_(capacity) {
}

SchemaCache&
SchemaCache::GetInstance() {
    static SchemaCache instance;
    return instance;
}

Status
SchemaCache::GetOrLoad(const std::string& endpoint, const std::string& db_name, const std::string& collection_name,
                       bool force_update, const void* load_scope, const Loader& loader, CollectionDescPtr& desc) {
    const auto key = CollectionCacheKey::Create(endpoint, db_name, collection_name);

    CollectionDescPtr initial_desc;
    if (getCached(key, initial_desc) && !force_update) {
        desc = std::move(initial_desc);
        return Status::OK();
    }

    if (load_scope == nullptr) {
        return {StatusCode::INVALID_ARGUMENT, "Schema cache load scope cannot be null"};
    }
    const LoadKey load_key{key, load_scope};

    LoadStatePtr load_state;
    bool is_loader = false;
    {
        std::lock_guard<std::mutex> lock(loading_mutex_);
        auto it = loading_.find(load_key);
        if (it == loading_.end()) {
            load_state = std::make_shared<LoadState>();
            loading_.emplace(load_key, load_state);
            is_loader = true;
        } else {
            load_state = it->second;
        }
    }

    if (!is_loader) {
        std::unique_lock<std::mutex> lock(load_state->mutex_);
        load_state->cv_.wait(lock, [&load_state]() { return load_state->completed_; });
        if (load_state->status_.IsOk()) {
            desc = load_state->desc_;
        }
        return load_state->status_;
    }

    auto finish_load = [this, &load_key, &load_state, &desc](const Status& status, const CollectionDescPtr& loaded) {
        {
            std::lock_guard<std::mutex> lock(load_state->mutex_);
            load_state->status_ = status;
            load_state->desc_ = loaded;
            load_state->completed_ = true;
        }
        load_state->cv_.notify_all();

        {
            std::lock_guard<std::mutex> lock(loading_mutex_);
            auto it = loading_.find(load_key);
            if (it != loading_.end() && it->second == load_state) {
                loading_.erase(it);
            }
        }

        if (status.IsOk()) {
            desc = loaded;
        }
        return status;
    };

    try {
        // A different loader might have populated the cache between the first lookup and
        // registration in loading_. For force refresh, a different cached pointer means
        // another refresh completed after this call's initial lookup.
        CollectionDescPtr current_desc;
        if (getCached(key, current_desc) && (!force_update || current_desc != initial_desc)) {
            return finish_load(Status::OK(), current_desc);
        }

        CollectionDescPtr loaded;
        auto status = loader(loaded);
        if (status.IsOk()) {
            std::unique_lock<std::shared_timed_mutex> lock(mutex_);
            // Do not repopulate an entry invalidated while the RPC was in flight.
            if (!load_state->invalidated_.load(std::memory_order_acquire)) {
                setCacheNoLocked(key, loaded);
            }
        }

        return finish_load(status, loaded);
    } catch (const std::exception& e) {
        return finish_load({StatusCode::UNKNOWN_ERROR, "Schema loader failed: " + std::string(e.what())}, nullptr);
    } catch (...) {
        return finish_load({StatusCode::UNKNOWN_ERROR, "Schema loader failed with unknown exception"}, nullptr);
    }
}

bool
SchemaCache::Get(const std::string& endpoint, const std::string& db_name, const std::string& collection_name,
                 CollectionDescPtr& desc) {
    return getCached(CollectionCacheKey::Create(endpoint, db_name, collection_name), desc);
}

void
SchemaCache::Set(const std::string& endpoint, const std::string& db_name, const std::string& collection_name,
                 CollectionDescPtr desc) {
    const auto key = CollectionCacheKey::Create(endpoint, db_name, collection_name);
    invalidateLoad(key);
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    setCacheNoLocked(key, std::move(desc));
}

void
SchemaCache::Invalidate(const std::string& endpoint, const std::string& db_name, const std::string& collection_name) {
    const auto key = CollectionCacheKey::Create(endpoint, db_name, collection_name);
    invalidateLoad(key);
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    cache_.erase(key);
}

void
SchemaCache::InvalidateDb(const std::string& endpoint, const std::string& db_name) {
    const auto prefix = CollectionCacheKey::Create(endpoint, db_name, "");
    invalidateDbLoads(prefix);
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
SchemaCache::Clear() {
    invalidateAllLoads();
    std::unique_lock<std::shared_timed_mutex> lock(mutex_);
    cache_.clear();
}

size_t
SchemaCache::Size() const {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    return cache_.size();
}

bool
SchemaCache::getCached(const CollectionCacheKey& key, CollectionDescPtr& desc) {
    std::shared_lock<std::shared_timed_mutex> lock(mutex_);
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        return false;
    }
    touch(it->second);
    desc = it->second->desc_;
    return true;
}

void
SchemaCache::setCacheNoLocked(const CollectionCacheKey& key, CollectionDescPtr desc) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        it->second->desc_ = std::move(desc);
        touch(it->second);
        return;
    }

    cache_.emplace(key, std::make_shared<Entry>(std::move(desc), nextAccess()));
    evictIfNeededLocked();
}

void
SchemaCache::invalidateLoad(const CollectionCacheKey& key) {
    std::lock_guard<std::mutex> lock(loading_mutex_);
    for (auto it = loading_.begin(); it != loading_.end();) {
        if (it->first.collection_key_ == key) {
            it->second->invalidated_.store(true, std::memory_order_release);
            it = loading_.erase(it);
        } else {
            ++it;
        }
    }
}

void
SchemaCache::invalidateDbLoads(const CollectionCacheKey& prefix) {
    std::lock_guard<std::mutex> lock(loading_mutex_);
    for (auto it = loading_.begin(); it != loading_.end();) {
        const auto& key = it->first.collection_key_;
        if (key.endpoint_ == prefix.endpoint_ && key.db_name_ == prefix.db_name_) {
            it->second->invalidated_.store(true, std::memory_order_release);
            it = loading_.erase(it);
        } else {
            ++it;
        }
    }
}

void
SchemaCache::invalidateAllLoads() {
    std::lock_guard<std::mutex> lock(loading_mutex_);
    for (const auto& pair : loading_) {
        pair.second->invalidated_.store(true, std::memory_order_release);
    }
    loading_.clear();
}

uint64_t
SchemaCache::nextAccess() {
    return access_sequence_.fetch_add(1, std::memory_order_relaxed) + 1;
}

void
SchemaCache::touch(const EntryPtr& entry) {
    const auto access = nextAccess();
    auto previous = entry->last_access_.load(std::memory_order_relaxed);
    while (previous < access && !entry->last_access_.compare_exchange_weak(previous, access, std::memory_order_relaxed,
                                                                           std::memory_order_relaxed)) {
    }
}

void
SchemaCache::evictIfNeededLocked() {
    if (cache_.size() <= capacity_) {
        return;
    }

    const auto oldest = std::min_element(cache_.begin(), cache_.end(), [](const auto& left, const auto& right) {
        return left.second->last_access_.load(std::memory_order_relaxed) <
               right.second->last_access_.load(std::memory_order_relaxed);
    });
    if (oldest != cache_.end()) {
        cache_.erase(oldest);
    }
}

}  // namespace milvus
