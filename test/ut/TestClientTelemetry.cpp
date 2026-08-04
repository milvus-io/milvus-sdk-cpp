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

    milvus::ClientRequestContext::Set("outer");
    {
        milvus::ScopedClientRequestId scoped("inner");
        EXPECT_EQ(milvus::ClientRequestContext::Get(), "inner");
    }
    EXPECT_EQ(milvus::ClientRequestContext::Get(), "outer");
    milvus::ClientRequestContext::Clear();
}
