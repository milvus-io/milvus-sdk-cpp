// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <milvus/thirdparty/nlohmann/json.hpp>
#include <thread>

#include "milvus.pb.h"
#include "milvus/ClientRequestContext.h"
#include "milvus/ClientTelemetry.h"

TEST(ClientTelemetryTest, MatchesCrossSdkConfigHashVector) {
    std::vector<milvus::TelemetryCommand> commands = {
        {"cfg-b", "push_config", "{\"sampling_rate\":0.5}", 0, true, ""},
        {"cfg-a", "push_config", "{\"heartbeat_interval_ms\":5000}", 0, true, ""},
    };
    EXPECT_EQ(milvus::ClientTelemetryManager::CalculateConfigHash(commands), "a271ff0bb1941777");
}

TEST(ClientTelemetryTest, RuntimeClientIdDoesNotBecomeStableConfiguration) {
    milvus::TelemetryConfig config;
    milvus::ClientTelemetryManager manager(config, "runtime-client-id");

    EXPECT_EQ(manager.ClientId(), "runtime-client-id");
    EXPECT_TRUE(manager.Config().client_id.empty());
}

TEST(ClientTelemetryTest, AppliesCommandsAndDeduplicatesIds) {
    milvus::TelemetryConfig config;
    config.enabled = false;
    milvus::ClientTelemetryManager manager(config);
    int calls = 0;
    manager.RegisterCommandHandler("custom", [&calls](const milvus::TelemetryCommand& command) {
        ++calls;
        return milvus::TelemetryCommandReply{command.command_id, true, "", ""};
    });

    manager.ProcessCommands({
        {"config", "push_config", "{\"heartbeat_interval_ms\":5000,\"sampling_rate\":0.25}", 1, true, ""},
        {"custom", "custom", "", 2, false, ""},
    });
    manager.ProcessCommands({{"custom", "custom", "", 2, false, ""}});

    EXPECT_EQ(manager.Config().heartbeat_interval_ms, 5000U);
    EXPECT_DOUBLE_EQ(manager.Config().sampling_rate, 0.25);
    EXPECT_EQ(manager.LastCommandTimestamp(), 2);
    EXPECT_FALSE(manager.ConfigHash().empty());
    EXPECT_EQ(calls, 1);
}

TEST(ClientTelemetryTest, ReconnectReuseMatchesOriginalUserConfig) {
    milvus::TelemetryConfig config;
    config.enabled = false;
    config.sampling_rate = 0.5;
    milvus::ClientTelemetryManager manager(config);

    manager.ProcessCommands({{"remote", "push_config", R"({"sampling_rate":0.25})", 1, true, ""}});
    EXPECT_DOUBLE_EQ(manager.Config().sampling_rate, 0.25);
    EXPECT_TRUE(manager.MatchesConnection(config, ""));

    auto changed = config;
    changed.enabled = true;
    EXPECT_FALSE(manager.MatchesConnection(changed, ""));
}

TEST(ClientTelemetryTest, PushConfigIsAtomicAndReportsAppliedAndIgnoredKeys) {
    milvus::TelemetryConfig config;
    config.enabled = false;
    milvus::ClientTelemetryManager manager(config);

    manager.ProcessCommands(
        {{"invalid", "push_config", R"({"enabled":true,"heartbeat_interval_ms":0})", 1, false, ""}});
    EXPECT_FALSE(manager.Config().enabled);
    ASSERT_EQ(manager.PendingCommandReplies().size(), 1U);
    EXPECT_FALSE(manager.PendingCommandReplies().back().success);

    manager.ProcessCommands(
        {{"valid", "push_config",
          R"({"unknown_b":1,"sampling_rate":2,"enabled":true,"ttl_seconds":3,"heartbeat_interval_ms":2500,"unknown_a":2})",
          2, false, ""}});
    auto updated = manager.Config();
    EXPECT_TRUE(updated.enabled);
    EXPECT_EQ(updated.heartbeat_interval_ms, 2500U);
    EXPECT_DOUBLE_EQ(updated.sampling_rate, 1.0);

    auto replies = manager.PendingCommandReplies();
    ASSERT_EQ(replies.size(), 2U);
    ASSERT_TRUE(replies.back().success);
    auto payload = nlohmann::json::parse(replies.back().payload);
    EXPECT_EQ(payload["applied"], nlohmann::json({"enabled", "heartbeat_interval_ms", "sampling_rate"}));
    EXPECT_EQ(payload["ignored"], nlohmann::json({"ttl_seconds", "unknown_a", "unknown_b"}));
}

TEST(ClientTelemetryTest, RejectsWrongCommandPayloadTypes) {
    milvus::TelemetryConfig config;
    config.enabled = false;
    milvus::ClientTelemetryManager manager(config);

    manager.ProcessCommands(
        {{"push", "push_config", R"({"enabled":"false"})", 1, false, ""},
         {"collection", "collection_metrics", R"({"enabled":false,"collections":"books"})", 2, false, ""},
         {"ttl", "push_config", R"({"enabled":true,"ttl_seconds":"bad"})", 3, false, ""}});

    auto replies = manager.PendingCommandReplies();
    ASSERT_EQ(replies.size(), 3U);
    EXPECT_FALSE(replies[0].success);
    EXPECT_FALSE(replies[1].success);
    EXPECT_FALSE(replies[2].success);
    EXPECT_FALSE(manager.Config().enabled);
}

TEST(ClientTelemetryTest, SerializesConcurrentCommandBatches) {
    milvus::TelemetryConfig config;
    config.enabled = false;
    milvus::ClientTelemetryManager manager(config);
    std::atomic<int> calls{0};
    std::atomic<bool> release{false};
    manager.RegisterCommandHandler("custom", [&calls, &release](const milvus::TelemetryCommand& command) {
        ++calls;
        while (!release.load()) {
            std::this_thread::yield();
        }
        return milvus::TelemetryCommandReply{command.command_id, true, "", ""};
    });
    const milvus::TelemetryCommand command{"same", "custom", "", 1, false, ""};

    std::thread first([&]() { manager.ProcessCommands({command}); });
    while (calls.load() == 0) {
        std::this_thread::yield();
    }
    std::thread second([&]() { manager.ProcessCommands({command}); });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    release = true;
    first.join();
    second.join();

    EXPECT_EQ(calls.load(), 1);
}

TEST(ClientTelemetryTest, UsesOneFixedPointSamplerAcrossOperations) {
    milvus::TelemetryConfig config;
    config.sampling_rate = 0.25;
    milvus::ClientTelemetryManager manager(config);
    milvus::proto::milvus::SearchRequest request;
    request.set_collection_name("books");

    for (int index = 0; index < 12; ++index) {
        manager.RecordOperation(index % 2 == 0 ? "Search" : "Query", request, std::chrono::steady_clock::now(), true,
                                "");
    }
    manager.Start();
    for (int retry = 0; retry < 100 && manager.MetricsSnapshots().empty(); ++retry) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    manager.Stop();

    auto snapshots = manager.MetricsSnapshots();
    ASSERT_FALSE(snapshots.empty());
    int64_t sampled = 0;
    for (const auto& operation : snapshots.back().metrics) {
        sampled += operation.global.request_count;
    }
    EXPECT_EQ(sampled, 3);
}

TEST(ClientTelemetryTest, StopAndRestartPreserveCommandState) {
    milvus::TelemetryConfig config;
    config.enabled = false;
    milvus::ClientTelemetryManager manager(config);
    int calls = 0;
    manager.RegisterCommandHandler("custom", [&calls](const milvus::TelemetryCommand& command) {
        ++calls;
        return milvus::TelemetryCommandReply{command.command_id, true, "", ""};
    });
    const milvus::TelemetryCommand command{"custom", "custom", "", 2, false, ""};

    manager.ProcessCommands({command});
    manager.Start();
    EXPECT_TRUE(manager.IsReady());
    manager.Stop();
    EXPECT_FALSE(manager.IsReady());
    manager.Start();
    EXPECT_TRUE(manager.IsReady());
    manager.ProcessCommands({command});

    EXPECT_EQ(calls, 1);
    EXPECT_EQ(manager.LastCommandTimestamp(), 2);
    manager.Stop();
}

TEST(ClientRequestContextTest, GeneratesAndScopesTraceIds) {
    auto request_id = milvus::ClientRequestContext::NewRequestId();
    EXPECT_EQ(request_id.size(), 32U);
    EXPECT_EQ(request_id.find_first_not_of("0123456789abcdef"), std::string::npos);
    EXPECT_NE(request_id, std::string(32, '0'));
    EXPECT_TRUE(milvus::ClientRequestContext::IsValid(request_id));
    EXPECT_FALSE(milvus::ClientRequestContext::IsValid(std::string(32, '0')));
    EXPECT_FALSE(milvus::ClientRequestContext::IsValid("ABCDEF0123456789ABCDEF0123456789"));
    EXPECT_FALSE(milvus::ClientRequestContext::IsValid("short"));

    milvus::ClientRequestContext::Set("outer");
    {
        milvus::ScopedClientRequestId scoped("inner");
        EXPECT_EQ(milvus::ClientRequestContext::Get(), "inner");
    }
    EXPECT_EQ(milvus::ClientRequestContext::Get(), "outer");
    milvus::ClientRequestContext::Clear();
}
