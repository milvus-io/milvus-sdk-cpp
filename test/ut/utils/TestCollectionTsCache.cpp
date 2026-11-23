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

#include <gtest/gtest.h>

#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "utils/TimeUtils.h"
#include "utils/cache/CollectionTsCache.h"

TEST(CollectionTsCacheTest, IsolatesEndpointDatabaseAndCollection) {
    milvus::CollectionTsCache cache;

    EXPECT_EQ(cache.Get("endpoint-a", "db", "collection"), 0);

    cache.Set("endpoint-a", "db", "collection", 100);
    cache.Set("endpoint-b", "db", "collection", 200);
    cache.Set("endpoint-a", "other-db", "collection", 300);
    cache.Set("endpoint-a", "db", "other-collection", 400);

    EXPECT_EQ(cache.Get("endpoint-a", "db", "collection"), 100);
    EXPECT_EQ(cache.Get("endpoint-b", "db", "collection"), 200);
    EXPECT_EQ(cache.Get("endpoint-a", "other-db", "collection"), 300);
    EXPECT_EQ(cache.Get("endpoint-a", "db", "other-collection"), 400);
}

TEST(CollectionTsCacheTest, NormalizesDefaultDatabaseAndKeepsTimestampMonotonic) {
    milvus::CollectionTsCache cache;

    cache.Set("http://localhost:19530/db-from-uri", "", "collection", 100);
    EXPECT_EQ(cache.Get("localhost:19530", "default", "collection"), 100);

    cache.Set("localhost:19530", "default", "collection", 99);
    cache.Set("localhost:19530", "default", "collection", 100);
    EXPECT_EQ(cache.Get("http://localhost:19530", "", "collection"), 100);

    cache.Set("localhost:19530", "default", "collection", 101);
    EXPECT_EQ(cache.Get("http://localhost:19530", "", "collection"), 101);

    cache.Set("localhost:19530", "default", "zero", 0);
    EXPECT_EQ(cache.Size(), 1);
}

TEST(CollectionTsCacheTest, MalformedEndpointFallsBackToRawValue) {
    milvus::CollectionTsCache cache;

    const std::string malformed_endpoint = "http://localhost:not-a-port";
    const std::string another_malformed_endpoint = "http://localhost:another-bad-port";

    EXPECT_NO_THROW(cache.Set(malformed_endpoint, "db", "collection", 100));
    EXPECT_EQ(cache.Get(malformed_endpoint, "db", "collection"), 100);
    EXPECT_EQ(cache.Get(another_malformed_endpoint, "db", "collection"), 0);
}

TEST(CollectionTsCacheTest, InvalidatesEntries) {
    milvus::CollectionTsCache cache;

    cache.Set("endpoint", "db", "first", 1);
    cache.Set("endpoint", "db", "second", 2);
    cache.Set("endpoint", "db", "third", 3);

    EXPECT_EQ(cache.Get("endpoint", "db", "first"), 1);
    EXPECT_EQ(cache.Get("endpoint", "db", "second"), 2);
    EXPECT_EQ(cache.Get("endpoint", "db", "third"), 3);

    cache.Invalidate("endpoint", "db", "first");
    EXPECT_EQ(cache.Get("endpoint", "db", "first"), 0);
    EXPECT_EQ(cache.Size(), 2);

    cache.Clear();
    EXPECT_EQ(cache.Size(), 0);
}

TEST(CollectionTsCacheTest, EvictsLeastRecentlyUsedEntriesAtCapacity) {
    milvus::CollectionTsCache cache(16);
    constexpr int kCollectionCount = 64;

    for (int i = 0; i < kCollectionCount; ++i) {
        cache.Set("endpoint", "db", "collection-" + std::to_string(i), static_cast<uint64_t>(i + 1));
    }

    EXPECT_EQ(cache.Size(), 16);
    // Entries written before the capacity was reached are the least recently
    // used once the cache fills, so the earliest keys are evicted.
    EXPECT_EQ(cache.Get("endpoint", "db", "collection-0"), 0);
    EXPECT_EQ(cache.Get("endpoint", "db", "collection-47"), 0);

    // Reading an entry refreshes its recency so it survives a later eviction.
    EXPECT_EQ(cache.Get("endpoint", "db", "collection-48"), 49);
    cache.Set("endpoint", "db", "collection-64", 65);
    EXPECT_EQ(cache.Get("endpoint", "db", "collection-48"), 49);
    EXPECT_EQ(cache.Size(), 16);
}

TEST(CollectionTsCacheTest, SetCapacityShrinksImmediately) {
    milvus::CollectionTsCache cache(32);
    for (int i = 0; i < 32; ++i) {
        cache.Set("endpoint", "db", "collection-" + std::to_string(i), static_cast<uint64_t>(i + 1));
    }
    EXPECT_EQ(cache.Size(), 32);

    cache.SetCapacity(8);
    EXPECT_EQ(cache.Size(), 8);
    EXPECT_EQ(cache.Capacity(), 8);

    cache.SetCapacity(64);
    EXPECT_EQ(cache.Capacity(), 64);
    for (int i = 0; i < 32; ++i) {
        cache.Set("endpoint", "db", "collection-" + std::to_string(i), static_cast<uint64_t>(i + 1));
    }
    EXPECT_EQ(cache.Size(), 32);
}

TEST(CollectionTsCacheTest, InvalidatesDatabase) {
    milvus::CollectionTsCache cache;

    cache.Set("endpoint", "db", "first", 1);
    cache.Set("endpoint", "db", "second", 2);
    cache.Set("endpoint", "other-db", "third", 3);
    cache.Set("other-endpoint", "db", "fourth", 4);

    cache.InvalidateDb("endpoint", "db");

    EXPECT_EQ(cache.Get("endpoint", "db", "first"), 0);
    EXPECT_EQ(cache.Get("endpoint", "db", "second"), 0);
    EXPECT_EQ(cache.Get("endpoint", "other-db", "third"), 3);
    EXPECT_EQ(cache.Get("other-endpoint", "db", "fourth"), 4);
}

TEST(CollectionTsCacheTest, MovesCollection) {
    milvus::CollectionTsCache cache;

    cache.Set("endpoint", "db", "old", 100);
    cache.Set("endpoint", "db", "new", 200);
    cache.Set("endpoint", "other-db", "old", 300);

    cache.Move("endpoint", "db", "old", "db", "new");

    EXPECT_EQ(cache.Get("endpoint", "db", "old"), 0);
    EXPECT_EQ(cache.Get("endpoint", "db", "new"), 200);
    EXPECT_EQ(cache.Get("endpoint", "other-db", "old"), 300);

    cache.Move("endpoint", "db", "missing", "db", "new");
    EXPECT_EQ(cache.Get("endpoint", "db", "new"), 200);
}

TEST(CollectionTsCacheTest, MovesCollectionAcrossDatabases) {
    milvus::CollectionTsCache cache;

    cache.Set("endpoint", "source-db", "old", 100);
    cache.Set("endpoint", "target-db", "new", 200);
    cache.Set("endpoint", "source-db", "new", 300);

    cache.Move("endpoint", "source-db", "old", "target-db", "new");

    EXPECT_EQ(cache.Get("endpoint", "source-db", "old"), 0);
    EXPECT_EQ(cache.Get("endpoint", "target-db", "new"), 200);
    EXPECT_EQ(cache.Get("endpoint", "source-db", "new"), 300);
}

TEST(CollectionTsCacheTest, CopiesCollectionWithoutRemovingSource) {
    milvus::CollectionTsCache cache;

    cache.Set("endpoint", "db", "collection", 100);
    cache.Set("endpoint", "db", "alias", 50);

    cache.Copy("endpoint", "db", "collection", "db", "alias");

    EXPECT_EQ(cache.Get("endpoint", "db", "collection"), 100);
    EXPECT_EQ(cache.Get("endpoint", "db", "alias"), 100);

    // Preserve a newer timestamp that might have been written through the alias
    // after the server applied the alias mutation but before its callback ran.
    cache.Set("endpoint", "db", "alias", 200);
    cache.Copy("endpoint", "db", "collection", "db", "alias");
    EXPECT_EQ(cache.Get("endpoint", "db", "collection"), 100);
    EXPECT_EQ(cache.Get("endpoint", "db", "alias"), 200);
}

TEST(CollectionTsCacheTest, TransferAtCapacityEvictsAndKeepsNewTarget) {
    milvus::CollectionTsCache cache(8);

    for (int i = 0; i < 8; ++i) {
        cache.Set("endpoint", "db", "collection-" + std::to_string(i), static_cast<uint64_t>(i + 1));
    }
    EXPECT_EQ(cache.Size(), 8);

    // Move is erase-then-insert: the source is removed before the target is
    // added, so the entry count never exceeds capacity and no eviction fires.
    // This asserts the transfer path keeps size bounded and carries the ts.
    cache.Move("endpoint", "db", "collection-0", "db", "moved");
    EXPECT_EQ(cache.Size(), 8);
    EXPECT_EQ(cache.Get("endpoint", "db", "collection-0"), 0);
    EXPECT_EQ(cache.Get("endpoint", "db", "moved"), 1);

    // Copy keeps the source, so when the target key is absent the insertion
    // pushes the count past capacity and triggers eviction. The source is the
    // LRU entry here (it has the oldest recency), so transfer() must touch it
    // before eviction runs or the copy would evict the source's own timestamp.
    cache.Copy("endpoint", "db", "collection-1", "db", "copied");
    EXPECT_EQ(cache.Size(), 8);
    EXPECT_EQ(cache.Get("endpoint", "db", "collection-1"), 2);
    EXPECT_EQ(cache.Get("endpoint", "db", "copied"), 2);
}

TEST(CollectionTsCacheTest, ConcurrentNewNameWriteBeforeMovePreservesTimestamp) {
    milvus::CollectionTsCache cache;
    cache.Set("endpoint", "db", "old", 100);

    std::mutex mutex;
    std::condition_variable cv;
    bool write_completed = false;

    std::thread write_thread([&]() {
        cache.Set("endpoint", "db", "new", 200);
        {
            std::lock_guard<std::mutex> lock(mutex);
            write_completed = true;
        }
        cv.notify_one();
    });
    std::thread rename_thread([&]() {
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&write_completed]() { return write_completed; });
        }
        cache.Move("endpoint", "db", "old", "db", "new");
    });

    write_thread.join();
    rename_thread.join();

    EXPECT_EQ(cache.Get("endpoint", "db", "old"), 0);
    EXPECT_EQ(cache.Get("endpoint", "db", "new"), 200);
}

TEST(CollectionTsCacheTest, MakeMktsFromNowMs) {
    const auto before = milvus::GetNowMs();
    const auto ts = milvus::MakeMktsFromNowMs();
    const auto after = milvus::GetNowMs();

    EXPECT_GE(ts, (before + 1000) << 18);
    EXPECT_LE(ts, (after + 1000) << 18);
}

TEST(CollectionTsCacheTest, ConcurrentReadsAndMonotonicWrites) {
    milvus::CollectionTsCache cache;
    constexpr int kThreadCount = 8;
    constexpr int kWritesPerThread = 1000;

    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (int thread_id = 0; thread_id < kThreadCount; ++thread_id) {
        threads.emplace_back([&cache, thread_id, kWritesPerThread]() {
            for (int i = 1; i <= kWritesPerThread; ++i) {
                const auto ts = static_cast<uint64_t>(thread_id * kWritesPerThread + i);
                cache.Set("endpoint", "db", "collection", ts);
                cache.Get("endpoint", "db", "collection");
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(cache.Get("endpoint", "db", "collection"), kThreadCount * kWritesPerThread);
}

TEST(CollectionTsCacheTest, ConcurrentSetsAndGetsAtCapacity) {
    milvus::CollectionTsCache cache(32);
    constexpr int kThreadCount = 8;
    constexpr int kKeysPerThread = 64;
    constexpr int kTotalKeys = kThreadCount * kKeysPerThread;

    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (int thread_id = 0; thread_id < kThreadCount; ++thread_id) {
        threads.emplace_back([&cache, thread_id, kKeysPerThread, kTotalKeys]() {
            for (int i = 0; i < kKeysPerThread; ++i) {
                const auto key = "collection-" + std::to_string(thread_id * kKeysPerThread + i);
                cache.Set("endpoint", "db", key, static_cast<uint64_t>(thread_id * kKeysPerThread + i + 1));
                cache.Get("endpoint", "db", key);
                // Re-write each key with a strictly larger ts so a surviving
                // entry must hold the last written value, not the first one.
                cache.Set("endpoint", "db", key,
                          static_cast<uint64_t>(thread_id * kKeysPerThread + i + 1 + kTotalKeys));
                cache.Get("endpoint", "db", key);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // The cache stays bounded at capacity under concurrent eviction.
    EXPECT_EQ(cache.Size(), 32);
    // A key that survives eviction must carry its last written (larger) ts; a
    // key that was evicted and never re-written reads back 0.
    for (int i = 0; i < kTotalKeys; ++i) {
        const auto key = "collection-" + std::to_string(i);
        const auto ts = cache.Get("endpoint", "db", key);
        const auto last_written = static_cast<uint64_t>(i + 1 + kTotalKeys);
        EXPECT_TRUE(ts == 0 || ts == last_written) << "key=" << key << " ts=" << ts;
    }
}
