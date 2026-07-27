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

TEST(CollectionTsCacheTest, DoesNotEvictSessionTimestamps) {
    milvus::CollectionTsCache cache;
    constexpr int kCollectionCount = 5000;

    for (int i = 0; i < kCollectionCount; ++i) {
        cache.Set("endpoint", "db", "collection-" + std::to_string(i), static_cast<uint64_t>(i + 1));
    }

    EXPECT_EQ(cache.Size(), kCollectionCount);
    EXPECT_EQ(cache.Get("endpoint", "db", "collection-0"), 1);
    EXPECT_EQ(cache.Get("endpoint", "db", "collection-4096"), 4097);
    EXPECT_EQ(cache.Get("endpoint", "db", "collection-4999"), 5000);
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
