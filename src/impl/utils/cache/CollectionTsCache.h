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
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "./CollectionCacheKey.h"

namespace milvus {

class CollectionTsCache {
 public:
    static CollectionTsCache&
    GetInstance();

    uint64_t
    Get(const std::string& endpoint, const std::string& db_name, const std::string& collection_name);

    void
    Set(const std::string& endpoint, const std::string& db_name, const std::string& collection_name, uint64_t ts);

    void
    Invalidate(const std::string& endpoint, const std::string& db_name, const std::string& collection_name);

    void
    InvalidateDb(const std::string& endpoint, const std::string& db_name);

    // Move the latest timestamp to a renamed collection and remove the old key.
    void
    Move(const std::string& endpoint, const std::string& old_db_name, const std::string& old_collection_name,
         const std::string& new_db_name, const std::string& new_collection_name);

    // Copy the latest timestamp to an alias while retaining the collection key. The target is
    // updated monotonically so a newer concurrent write through the alias is not overwritten.
    void
    Copy(const std::string& endpoint, const std::string& source_db_name, const std::string& source_collection_name,
         const std::string& target_db_name, const std::string& target_collection_name);

    void
    Clear();

    size_t
    Size() const;

 private:
    void
    transfer(const CollectionCacheKey& source_key, const CollectionCacheKey& target_key, bool drop_source);

    mutable std::shared_timed_mutex mutex_;
    std::unordered_map<CollectionCacheKey, uint64_t, CollectionCacheKeyHash> cache_;
};

}  // namespace milvus
