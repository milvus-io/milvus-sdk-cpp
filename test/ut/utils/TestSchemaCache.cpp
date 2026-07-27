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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "utils/cache/SchemaCache.h"

namespace {

milvus::CollectionDescPtr
MakeCollectionDesc(int64_t id) {
    auto desc = std::make_shared<milvus::CollectionDesc>();
    desc->SetID(id);
    return desc;
}

const void*
TestLoadScope() {
    static const int scope = 0;
    return &scope;
}

}  // namespace

TEST(SchemaCacheTest, IsolatesEndpointDatabaseAndCollection) {
    milvus::SchemaCache cache;

    cache.Set("endpoint-a", "db", "collection", MakeCollectionDesc(1));
    cache.Set("endpoint-b", "db", "collection", MakeCollectionDesc(2));
    cache.Set("endpoint-a", "other-db", "collection", MakeCollectionDesc(3));
    cache.Set("endpoint-a", "db", "other-collection", MakeCollectionDesc(4));

    milvus::CollectionDescPtr desc;
    ASSERT_TRUE(cache.Get("endpoint-a", "db", "collection", desc));
    EXPECT_EQ(desc->ID(), 1);
    ASSERT_TRUE(cache.Get("endpoint-b", "db", "collection", desc));
    EXPECT_EQ(desc->ID(), 2);
    ASSERT_TRUE(cache.Get("endpoint-a", "other-db", "collection", desc));
    EXPECT_EQ(desc->ID(), 3);
    ASSERT_TRUE(cache.Get("endpoint-a", "db", "other-collection", desc));
    EXPECT_EQ(desc->ID(), 4);
}

TEST(SchemaCacheTest, NormalizesDefaultDatabaseAndInvalidatesDatabase) {
    milvus::SchemaCache cache;
    cache.Set("http://localhost:19530/db-from-uri", "", "first", MakeCollectionDesc(1));
    cache.Set("localhost:19530", "default", "second", MakeCollectionDesc(2));
    cache.Set("localhost:19530", "other", "third", MakeCollectionDesc(3));
    cache.Set("other-endpoint", "default", "fourth", MakeCollectionDesc(4));

    milvus::CollectionDescPtr desc;
    ASSERT_TRUE(cache.Get("localhost:19530", "default", "first", desc));
    EXPECT_EQ(desc->ID(), 1);

    cache.InvalidateDb("http://localhost:19530", "");
    EXPECT_FALSE(cache.Get("localhost:19530", "default", "first", desc));
    EXPECT_FALSE(cache.Get("localhost:19530", "", "second", desc));
    EXPECT_TRUE(cache.Get("localhost:19530", "other", "third", desc));
    EXPECT_TRUE(cache.Get("other-endpoint", "default", "fourth", desc));
}

TEST(SchemaCacheTest, LoadsCachesAndForceRefreshes) {
    milvus::SchemaCache cache;
    int load_count = 0;
    auto loader = [&load_count](milvus::CollectionDescPtr& desc) {
        desc = MakeCollectionDesc(++load_count);
        return milvus::Status::OK();
    };

    milvus::CollectionDescPtr desc;
    auto status = cache.GetOrLoad("endpoint", "db", "collection", false, TestLoadScope(), loader, desc);
    ASSERT_TRUE(status.IsOk());
    EXPECT_EQ(desc->ID(), 1);

    status = cache.GetOrLoad("endpoint", "db", "collection", false, TestLoadScope(), loader, desc);
    ASSERT_TRUE(status.IsOk());
    EXPECT_EQ(desc->ID(), 1);
    EXPECT_EQ(load_count, 1);

    status = cache.GetOrLoad("endpoint", "db", "collection", true, TestLoadScope(), loader, desc);
    ASSERT_TRUE(status.IsOk());
    EXPECT_EQ(desc->ID(), 2);
    EXPECT_EQ(load_count, 2);
}

TEST(SchemaCacheTest, DoesNotCacheLoaderFailure) {
    milvus::SchemaCache cache;
    int load_count = 0;
    auto loader = [&load_count](milvus::CollectionDescPtr&) {
        ++load_count;
        return milvus::Status{milvus::StatusCode::SERVER_FAILED, "load failed"};
    };

    milvus::CollectionDescPtr desc;
    auto status = cache.GetOrLoad("endpoint", "db", "collection", false, TestLoadScope(), loader, desc);
    EXPECT_FALSE(status.IsOk());
    EXPECT_FALSE(cache.Get("endpoint", "db", "collection", desc));

    status = cache.GetOrLoad("endpoint", "db", "collection", false, TestLoadScope(), loader, desc);
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(load_count, 2);
}

TEST(SchemaCacheTest, ThrowingLoaderReleasesConcurrentWaiterAndAllowsRetry) {
    milvus::SchemaCache cache;
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool loader_started = false;
    bool allow_loader_to_throw = false;
    std::atomic<int> load_count{0};

    auto throwing_loader = [&](milvus::CollectionDescPtr&) -> milvus::Status {
        load_count.fetch_add(1);
        std::unique_lock<std::mutex> lock(gate_mutex);
        loader_started = true;
        gate_cv.notify_all();
        gate_cv.wait(lock, [&allow_loader_to_throw]() { return allow_loader_to_throw; });
        throw std::runtime_error("malformed schema");
    };

    milvus::CollectionDescPtr first_desc;
    auto first = std::async(std::launch::async, [&]() {
        return cache.GetOrLoad("endpoint", "db", "collection", false, TestLoadScope(), throwing_loader, first_desc);
    });

    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        gate_cv.wait(lock, [&loader_started]() { return loader_started; });
    }

    std::promise<void> waiter_entered;
    auto waiter_started = waiter_entered.get_future();
    milvus::CollectionDescPtr waiter_desc;
    auto waiter = std::async(std::launch::async, [&]() {
        waiter_entered.set_value();
        return cache.GetOrLoad("endpoint", "db", "collection", false, TestLoadScope(), throwing_loader, waiter_desc);
    });
    waiter_started.wait();
    EXPECT_EQ(waiter.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout);

    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        allow_loader_to_throw = true;
    }
    gate_cv.notify_all();

    auto first_status = first.get();
    auto waiter_status = waiter.get();
    EXPECT_FALSE(first_status.IsOk());
    EXPECT_FALSE(waiter_status.IsOk());
    EXPECT_NE(first_status.Message().find("malformed schema"), std::string::npos);
    EXPECT_EQ(waiter_status.Message(), first_status.Message());
    EXPECT_EQ(load_count.load(), 1);

    auto recovery_loader = [](milvus::CollectionDescPtr& desc) {
        desc = MakeCollectionDesc(200);
        return milvus::Status::OK();
    };
    milvus::CollectionDescPtr recovered;
    auto recovery_status =
        cache.GetOrLoad("endpoint", "db", "collection", false, TestLoadScope(), recovery_loader, recovered);
    ASSERT_TRUE(recovery_status.IsOk());
    ASSERT_NE(recovered, nullptr);
    EXPECT_EQ(recovered->ID(), 200);
}

TEST(SchemaCacheTest, InvalidatesAndEvictsLeastRecentlyUsedEntry) {
    milvus::SchemaCache cache(2);
    cache.Set("endpoint", "db", "first", MakeCollectionDesc(1));
    cache.Set("endpoint", "db", "second", MakeCollectionDesc(2));

    milvus::CollectionDescPtr desc;
    ASSERT_TRUE(cache.Get("endpoint", "db", "first", desc));
    cache.Set("endpoint", "db", "third", MakeCollectionDesc(3));

    EXPECT_FALSE(cache.Get("endpoint", "db", "second", desc));
    EXPECT_TRUE(cache.Get("endpoint", "db", "first", desc));
    EXPECT_TRUE(cache.Get("endpoint", "db", "third", desc));

    cache.Invalidate("endpoint", "db", "first");
    EXPECT_FALSE(cache.Get("endpoint", "db", "first", desc));
    EXPECT_EQ(cache.Size(), 1);

    cache.Clear();
    EXPECT_EQ(cache.Size(), 0);
}

TEST(SchemaCacheTest, ConcurrentMissesShareOneLoad) {
    milvus::SchemaCache cache;
    constexpr int kThreadCount = 8;
    std::atomic<int> entered{0};
    std::atomic<int> load_count{0};
    std::mutex gate_mutex;
    std::condition_variable gate_cv;

    auto loader = [&](milvus::CollectionDescPtr& desc) {
        load_count.fetch_add(1);
        std::unique_lock<std::mutex> lock(gate_mutex);
        gate_cv.wait(lock, [&entered, kThreadCount]() { return entered.load() == kThreadCount; });
        desc = MakeCollectionDesc(100);
        return milvus::Status::OK();
    };

    std::vector<milvus::Status> statuses(kThreadCount);
    std::vector<milvus::CollectionDescPtr> descs(kThreadCount);
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (int i = 0; i < kThreadCount; ++i) {
        threads.emplace_back([&, i]() {
            entered.fetch_add(1);
            gate_cv.notify_all();
            statuses[i] = cache.GetOrLoad("endpoint", "db", "collection", false, TestLoadScope(), loader, descs[i]);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(load_count.load(), 1);
    for (int i = 0; i < kThreadCount; ++i) {
        EXPECT_TRUE(statuses[i].IsOk());
        ASSERT_NE(descs[i], nullptr);
        EXPECT_EQ(descs[i]->ID(), 100);
    }
}

TEST(SchemaCacheTest, DifferentLoadScopesDoNotShareInFlightLoad) {
    milvus::SchemaCache cache;
    int first_scope = 0;
    int second_scope = 0;
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool first_loader_started = false;
    bool allow_first_loader_to_finish = false;
    std::atomic<bool> second_loader_started{false};

    auto first_loader = [&](milvus::CollectionDescPtr& desc) {
        std::unique_lock<std::mutex> lock(gate_mutex);
        first_loader_started = true;
        gate_cv.notify_all();
        gate_cv.wait(lock, [&allow_first_loader_to_finish]() { return allow_first_loader_to_finish; });
        desc = MakeCollectionDesc(100);
        return milvus::Status::OK();
    };
    auto second_loader = [&](milvus::CollectionDescPtr& desc) {
        second_loader_started.store(true);
        desc = MakeCollectionDesc(200);
        return milvus::Status::OK();
    };

    milvus::Status first_status;
    milvus::CollectionDescPtr first_desc;
    std::thread first_thread([&]() {
        first_status = cache.GetOrLoad("endpoint", "db", "collection", false, &first_scope, first_loader, first_desc);
    });

    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        gate_cv.wait(lock, [&first_loader_started]() { return first_loader_started; });
    }

    milvus::CollectionDescPtr second_desc;
    auto second = std::async(std::launch::async, [&]() {
        return cache.GetOrLoad("endpoint", "db", "collection", false, &second_scope, second_loader, second_desc);
    });
    const bool second_completed_independently = second.wait_for(std::chrono::seconds(1)) == std::future_status::ready;

    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        allow_first_loader_to_finish = true;
    }
    gate_cv.notify_all();

    auto second_status = second.get();
    first_thread.join();

    EXPECT_TRUE(second_completed_independently);
    EXPECT_TRUE(second_loader_started.load());
    EXPECT_TRUE(first_status.IsOk());
    EXPECT_TRUE(second_status.IsOk());
    ASSERT_NE(first_desc, nullptr);
    ASSERT_NE(second_desc, nullptr);
    EXPECT_EQ(first_desc->ID(), 100);
    EXPECT_EQ(second_desc->ID(), 200);
}

TEST(SchemaCacheTest, InvalidateDuringLoadPreventsCacheRepopulation) {
    milvus::SchemaCache cache;
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool loader_started = false;
    bool allow_loader_to_finish = false;

    auto loader = [&](milvus::CollectionDescPtr& desc) {
        std::unique_lock<std::mutex> lock(gate_mutex);
        loader_started = true;
        gate_cv.notify_all();
        gate_cv.wait(lock, [&allow_loader_to_finish]() { return allow_loader_to_finish; });
        desc = MakeCollectionDesc(100);
        return milvus::Status::OK();
    };

    milvus::Status status;
    milvus::CollectionDescPtr loaded;
    std::thread thread(
        [&]() { status = cache.GetOrLoad("endpoint", "db", "collection", false, TestLoadScope(), loader, loaded); });

    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        gate_cv.wait(lock, [&loader_started]() { return loader_started; });
    }
    cache.Invalidate("endpoint", "db", "collection");
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        allow_loader_to_finish = true;
    }
    gate_cv.notify_all();
    thread.join();

    EXPECT_TRUE(status.IsOk());
    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->ID(), 100);

    milvus::CollectionDescPtr cached;
    EXPECT_FALSE(cache.Get("endpoint", "db", "collection", cached));
}

TEST(SchemaCacheTest, CallerAfterInvalidationStartsFreshLoad) {
    milvus::SchemaCache cache;
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool first_loader_started = false;
    bool second_loader_started = false;
    bool allow_first_loader_to_finish = false;

    auto first_loader = [&](milvus::CollectionDescPtr& desc) {
        std::unique_lock<std::mutex> lock(gate_mutex);
        first_loader_started = true;
        gate_cv.notify_all();
        gate_cv.wait(lock, [&allow_first_loader_to_finish]() { return allow_first_loader_to_finish; });
        desc = MakeCollectionDesc(100);
        return milvus::Status::OK();
    };
    auto second_loader = [&](milvus::CollectionDescPtr& desc) {
        {
            std::lock_guard<std::mutex> lock(gate_mutex);
            second_loader_started = true;
        }
        gate_cv.notify_all();
        desc = MakeCollectionDesc(200);
        return milvus::Status::OK();
    };

    milvus::Status first_status;
    milvus::Status second_status;
    milvus::CollectionDescPtr first_desc;
    milvus::CollectionDescPtr second_desc;
    std::thread first_thread([&]() {
        first_status =
            cache.GetOrLoad("endpoint", "db", "collection", false, TestLoadScope(), first_loader, first_desc);
    });

    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        gate_cv.wait(lock, [&first_loader_started]() { return first_loader_started; });
    }
    cache.Invalidate("endpoint", "db", "collection");

    std::thread second_thread([&]() {
        second_status =
            cache.GetOrLoad("endpoint", "db", "collection", false, TestLoadScope(), second_loader, second_desc);
    });

    bool fresh_load_started = false;
    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        fresh_load_started = gate_cv.wait_for(lock, std::chrono::seconds(1),
                                              [&second_loader_started]() { return second_loader_started; });
        allow_first_loader_to_finish = true;
    }
    gate_cv.notify_all();

    first_thread.join();
    second_thread.join();

    EXPECT_TRUE(fresh_load_started);
    EXPECT_TRUE(first_status.IsOk());
    EXPECT_TRUE(second_status.IsOk());
    ASSERT_NE(first_desc, nullptr);
    ASSERT_NE(second_desc, nullptr);
    EXPECT_EQ(first_desc->ID(), 100);
    EXPECT_EQ(second_desc->ID(), 200);

    milvus::CollectionDescPtr cached;
    ASSERT_TRUE(cache.Get("endpoint", "db", "collection", cached));
    EXPECT_EQ(cached->ID(), 200);
}

TEST(SchemaCacheTest, UnrelatedInvalidationDoesNotSuppressLoad) {
    milvus::SchemaCache cache;
    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool loader_started = false;
    bool allow_loader_to_finish = false;

    auto loader = [&](milvus::CollectionDescPtr& desc) {
        std::unique_lock<std::mutex> lock(gate_mutex);
        loader_started = true;
        gate_cv.notify_all();
        gate_cv.wait(lock, [&allow_loader_to_finish]() { return allow_loader_to_finish; });
        desc = MakeCollectionDesc(100);
        return milvus::Status::OK();
    };

    milvus::Status status;
    milvus::CollectionDescPtr loaded;
    std::thread thread(
        [&]() { status = cache.GetOrLoad("endpoint", "db", "first", false, TestLoadScope(), loader, loaded); });

    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        gate_cv.wait(lock, [&loader_started]() { return loader_started; });
    }
    cache.Invalidate("endpoint", "db", "second");
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        allow_loader_to_finish = true;
    }
    gate_cv.notify_all();
    thread.join();

    EXPECT_TRUE(status.IsOk());
    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->ID(), 100);

    milvus::CollectionDescPtr cached;
    ASSERT_TRUE(cache.Get("endpoint", "db", "first", cached));
    EXPECT_EQ(cached->ID(), 100);
}
