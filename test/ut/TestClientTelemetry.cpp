// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

#include <grpcpp/create_channel.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iomanip>
#include <milvus/thirdparty/nlohmann/json.hpp>
#include <sstream>
#include <thread>

#include "MilvusConnection.h"
#include "milvus.grpc.pb.h"
#include "milvus.pb.h"
#include "milvus/ClientRequestContext.h"
#include "milvus/ClientTelemetry.h"
#include "milvus/MilvusClient.h"
#include "milvus/MilvusClientV2.h"

namespace {

std::string
Rfc3339(std::chrono::system_clock::time_point value) {
    auto raw = std::chrono::system_clock::to_time_t(value);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &raw);
#else
    gmtime_r(&raw, &utc);
#endif
    std::ostringstream stream;
    stream << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

class ReconnectTelemetryService final : public milvus::proto::milvus::ClientTelemetryService::Service {
 public:
    grpc::Status
    ClientHeartbeat(grpc::ServerContext*, const milvus::proto::milvus::ClientHeartbeatRequest*,
                    milvus::proto::milvus::ClientHeartbeatResponse* response) override {
        ++heartbeats;
        auto* command = response->add_commands();
        command->set_command_id("reconnect");
        command->set_command_type("reconnect");
        command->set_create_time(1);
        return grpc::Status::OK;
    }

    std::atomic<int> heartbeats{0};
};

class ControlPlaneTelemetryService final : public milvus::proto::milvus::ClientTelemetryService::Service {
 public:
    grpc::Status
    ClientHeartbeat(grpc::ServerContext*, const milvus::proto::milvus::ClientHeartbeatRequest* request,
                    milvus::proto::milvus::ClientHeartbeatResponse* response) override {
        std::unique_lock<std::mutex> lock(mutex_);
        requests_.push_back(*request);
        const auto heartbeat = requests_.size();
        condition_.notify_all();
        if (heartbeat == 1) {
            auto* command = response->add_commands();
            command->set_command_id("disable");
            command->set_command_type("push_config");
            command->set_payload(R"({"enabled":false})");
            command->set_create_time(1);
            command->set_persistent(true);
        } else if (heartbeat == 2) {
            condition_.wait(lock, [this]() { return allow_enable_; });
            auto* command = response->add_commands();
            command->set_command_id("enable");
            command->set_command_type("push_config");
            command->set_payload(R"({"enabled":true})");
            command->set_create_time(2);
            command->set_persistent(true);
        }
        return grpc::Status::OK;
    }

    bool
    WaitForHeartbeats(size_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        return condition_.wait_for(lock, std::chrono::seconds(2),
                                   [this, count]() { return requests_.size() >= count; });
    }

    void
    AllowEnable() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            allow_enable_ = true;
        }
        condition_.notify_all();
    }

    std::vector<milvus::proto::milvus::ClientHeartbeatRequest>
    Requests() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return requests_;
    }

    size_t
    RequestCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return requests_.size();
    }

 private:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    bool allow_enable_{false};
    std::vector<milvus::proto::milvus::ClientHeartbeatRequest> requests_;
};

class LifecycleTelemetryService final : public milvus::proto::milvus::MilvusService::Service,
                                        public milvus::proto::milvus::ClientTelemetryService::Service {
 public:
    explicit LifecycleTelemetryService(std::string command_type) : command_type_(std::move(command_type)) {
    }

    grpc::Status
    Connect(grpc::ServerContext*, const milvus::proto::milvus::ConnectRequest*,
            milvus::proto::milvus::ConnectResponse* response) override {
        {
            std::lock_guard<std::mutex> lock(connect_mutex_);
            ++connects;
            if (fail_next_connect_) {
                fail_next_connect_ = false;
                response->mutable_status()->set_code(1);
                response->mutable_status()->set_reason("rejected connect");
            }
        }
        connect_condition_.notify_all();
        return grpc::Status::OK;
    }

    grpc::Status
    HasCollection(grpc::ServerContext*, const milvus::proto::milvus::HasCollectionRequest*,
                  milvus::proto::milvus::BoolResponse*) override {
        ++has_collections;
        return grpc::Status::OK;
    }

    grpc::Status
    ClientHeartbeat(grpc::ServerContext*, const milvus::proto::milvus::ClientHeartbeatRequest* request,
                    milvus::proto::milvus::ClientHeartbeatResponse* response) override {
        ++heartbeats;
        const auto database = request->client_info().reserved().find("db_name");
        if (database != request->client_info().reserved().end() && database->second == "secondary") {
            saw_secondary_database = true;
        }
        for (const auto& reply : request->command_replies()) {
            if (reply.command_id() == command_type_ && !reply.success()) {
                saw_failed_command_reply = true;
            }
        }
        if (commands_enabled.load() && !command_sent.exchange(true)) {
            auto* command = response->add_commands();
            command->set_command_id(command_type_);
            command->set_command_type(command_type_);
            command->set_create_time(1);
        }
        return grpc::Status::OK;
    }

    std::atomic<int> connects{0};
    std::atomic<int> heartbeats{0};
    std::atomic<int> has_collections{0};
    std::atomic<bool> saw_secondary_database{false};
    std::atomic<bool> saw_failed_command_reply{false};

    void
    EnableCommands() {
        commands_enabled = true;
    }

    void
    FailNextConnect() {
        std::lock_guard<std::mutex> lock(connect_mutex_);
        fail_next_connect_ = true;
    }

    bool
    WaitForConnects(int count) {
        std::unique_lock<std::mutex> lock(connect_mutex_);
        return connect_condition_.wait_for(lock, std::chrono::seconds(2),
                                           [this, count]() { return connects.load() >= count; });
    }

 private:
    std::string command_type_;
    std::atomic<bool> commands_enabled{false};
    std::atomic<bool> command_sent{false};
    std::mutex connect_mutex_;
    std::condition_variable connect_condition_;
    bool fail_next_connect_{false};
};

class RequestMetadataService final : public milvus::proto::milvus::MilvusService::Service {
 public:
    grpc::Status
    Connect(grpc::ServerContext*, const milvus::proto::milvus::ConnectRequest*,
            milvus::proto::milvus::ConnectResponse*) override {
        return grpc::Status::OK;
    }

    grpc::Status
    HasCollection(grpc::ServerContext* context, const milvus::proto::milvus::HasCollectionRequest*,
                  milvus::proto::milvus::BoolResponse*) override {
        const auto& metadata = context->client_metadata();
        const auto request_id = metadata.find("client_request_id");
        std::lock_guard<std::mutex> lock(mutex_);
        request_ids_.emplace_back(request_id != metadata.end(),
                                  request_id == metadata.end()
                                      ? std::string{}
                                      : std::string(request_id->second.data(), request_id->second.size()));
        return grpc::Status::OK;
    }

    std::vector<std::pair<bool, std::string>>
    RequestIds() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return request_ids_;
    }

 private:
    mutable std::mutex mutex_;
    std::vector<std::pair<bool, std::string>> request_ids_;
};

class DestructionSignal final {
 public:
    explicit DestructionSignal(std::shared_ptr<std::promise<void>> signal) : signal_(std::move(signal)) {
    }

    ~DestructionSignal() {
        signal_->set_value();
    }

 private:
    std::shared_ptr<std::promise<void>> signal_;
};

std::unique_ptr<grpc::Server>
StartLifecycleServer(LifecycleTelemetryService& service, int& port) {
    grpc::ServerBuilder builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(static_cast<milvus::proto::milvus::MilvusService::Service*>(&service));
    builder.RegisterService(static_cast<milvus::proto::milvus::ClientTelemetryService::Service*>(&service));
    return builder.BuildAndStart();
}

milvus::ConnectParam
TelemetryConnectParam(int port) {
    milvus::ConnectParam param("http://127.0.0.1:" + std::to_string(port));
    milvus::TelemetryConfig config;
    config.heartbeat_interval_ms = 1;
    param.SetTelemetryConfig(config);
    return param;
}

}  // namespace

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
    manager.ProcessCommands({{"custom", "custom", "", 2, false, ""}});

    EXPECT_EQ(manager.Config().heartbeat_interval_ms, 5000U);
    EXPECT_DOUBLE_EQ(manager.Config().sampling_rate, 0.25);
    EXPECT_EQ(manager.LastCommandTimestamp(), 2);
    EXPECT_FALSE(manager.ConfigHash().empty());
    EXPECT_EQ(calls, 1);
}

TEST(ClientTelemetryTest, RetainsMoreThanOneHundredTwentySnapshotsWithinOneHour) {
    milvus::TelemetryConfig config;
    milvus::ClientTelemetryManager manager(config);
    milvus::proto::milvus::SearchRequest request;

    constexpr size_t expected_snapshots = 121;
    for (size_t index = 0; index < expected_snapshots; ++index) {
        manager.RecordOperation("Search", request, std::chrono::steady_clock::now(), true, "");
        manager.Start();
        for (int retry = 0; retry < 1000 && manager.MetricsSnapshots().size() <= index; ++retry) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        manager.Stop();
        ASSERT_GT(manager.MetricsSnapshots().size(), index);
    }

    EXPECT_EQ(manager.MetricsSnapshots().size(), expected_snapshots);
}

TEST(ClientTelemetryTest, RetainsOneSecondHeartbeatWindowAndSkipsEmptyIntervals) {
    milvus::TelemetryConfig config;
    config.heartbeat_interval_ms = 1;
    milvus::ClientTelemetryManager manager(config);
    milvus::proto::milvus::SearchRequest request;

    manager.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    manager.Stop();
    EXPECT_TRUE(manager.MetricsSnapshots().empty());

    constexpr size_t generated_snapshots = 4104;
    constexpr size_t retained_snapshots = 4096;
    for (size_t index = 0; index < generated_snapshots; ++index) {
        manager.RecordOperation("Search", request, std::chrono::steady_clock::now(), true, "");
        manager.Start();
        manager.Stop();
    }

    EXPECT_EQ(manager.MetricsSnapshots().size(), retained_snapshots);
}

TEST(ClientTelemetryTest, AggregatesP99FromRetainedLatencySamples) {
    milvus::TelemetryConfig config;
    milvus::ClientTelemetryManager manager(config);
    milvus::proto::milvus::SearchRequest request;

    for (int index = 0; index < 100; ++index) {
        manager.RecordOperation("Search", request, std::chrono::steady_clock::now() - std::chrono::milliseconds(1),
                                true, "");
    }
    manager.Start();
    for (int retry = 0; retry < 1000 && manager.MetricsSnapshots().empty(); ++retry) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    manager.Stop();

    for (int index = 0; index < 100; ++index) {
        manager.RecordOperation("Search", request, std::chrono::steady_clock::now() - std::chrono::milliseconds(100),
                                true, "");
    }
    manager.Start();
    for (int retry = 0; retry < 1000 && manager.MetricsSnapshots().size() < 2; ++retry) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    manager.Stop();

    const auto now = std::chrono::system_clock::now();
    const auto payload = nlohmann::json{
        {"start_time", Rfc3339(now - std::chrono::minutes(1))},
        {"end_time", Rfc3339(now + std::chrono::minutes(1))},
        {"detail", false}}.dump();
    manager.ProcessCommands({{"history", "show_latency_history", payload, 1, false, ""}});

    const auto replies = manager.PendingCommandReplies();
    ASSERT_FALSE(replies.empty());
    ASSERT_TRUE(replies.back().success) << replies.back().error_message;
    const auto response = nlohmann::json::parse(replies.back().payload);
    EXPECT_GT(response["aggregated"]["metrics"]["Search"]["p99_latency_ms"].get<double>(), 90.0);
}

TEST(ClientTelemetryTest, CompressedQuantileHistoryPreservesEndpointsAndSlowTail) {
    milvus::TelemetryConfig config;
    milvus::ClientTelemetryManager manager(config);
    milvus::proto::milvus::SearchRequest request;

    // The exact per-window p99 is still in the fast group (indices 0..990), while
    // the 128-point history compression must include the slow endpoint/tail that
    // starts at evenly-spaced source index 991.
    for (int index = 0; index < 991; ++index) {
        manager.RecordOperation("Search", request, std::chrono::steady_clock::now() - std::chrono::milliseconds(1),
                                true, "");
    }
    for (int index = 0; index < 9; ++index) {
        manager.RecordOperation("Search", request, std::chrono::steady_clock::now() - std::chrono::milliseconds(250),
                                true, "");
    }
    manager.Start();
    for (int retry = 0; retry < 1000 && manager.MetricsSnapshots().empty(); ++retry) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    manager.Stop();

    const auto snapshots = manager.MetricsSnapshots();
    ASSERT_EQ(snapshots.size(), 1U);
    ASSERT_EQ(snapshots.front().metrics.size(), 1U);
    EXPECT_LT(snapshots.front().metrics.front().global.p99_latency_ms, 50.0);
    EXPECT_GT(snapshots.front().metrics.front().global.max_latency_ms, 200.0);

    const auto now = std::chrono::system_clock::now();
    const auto payload = nlohmann::json{
        {"start_time", Rfc3339(now - std::chrono::minutes(1))},
        {"end_time", Rfc3339(now + std::chrono::minutes(1))},
        {"detail", false}}.dump();
    manager.ProcessCommands({{"compressed-history", "show_latency_history", payload, 1, false, ""}});

    const auto replies = manager.PendingCommandReplies();
    ASSERT_FALSE(replies.empty());
    ASSERT_TRUE(replies.back().success) << replies.back().error_message;
    const auto response = nlohmann::json::parse(replies.back().payload);
    const auto metric = response["aggregated"]["metrics"]["Search"];
    EXPECT_GT(metric["p99_latency_ms"].get<double>(), 200.0);
    EXPECT_GT(metric["max_latency_ms"].get<double>(), 200.0);
}

TEST(ClientTelemetryTest, ReusesHeartbeatWorkerWhenReconnectRunsInCommandHandler) {
    ReconnectTelemetryService service;
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    ASSERT_NE(server, nullptr);

    std::atomic<int> reconnects{0};
    {
        milvus::TelemetryConfig config;
        config.heartbeat_interval_ms = 1;
        milvus::ClientTelemetryManager manager(config);
        manager.AttachChannel(
            grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()), "", "", "",
            "", "");
        manager.RegisterCommandHandler("reconnect", [&manager, &reconnects](const milvus::TelemetryCommand& command) {
            ++reconnects;
            manager.Stop();
            manager.Start();
            return milvus::TelemetryCommandReply{command.command_id, true, "", ""};
        });

        manager.Start();
        for (int retry = 0; retry < 2000 && service.heartbeats.load() < 2; ++retry) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        manager.Stop();
    }
    server->Shutdown();

    EXPECT_GE(service.heartbeats.load(), 2);
    EXPECT_EQ(reconnects.load(), 1);
}

TEST(ClientTelemetryTest, ExternalStopWinsRaceWithHeartbeatSelfRestart) {
    ReconnectTelemetryService service;
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    ASSERT_NE(server, nullptr);

    milvus::TelemetryConfig config;
    config.heartbeat_interval_ms = 1;
    milvus::ClientTelemetryManager manager(config);
    manager.AttachChannel(grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()),
                          "", "", "", "", "");

    std::atomic<bool> handler_entered{false};
    std::atomic<bool> allow_self_restart{false};
    std::atomic<bool> self_restart_ready{true};
    manager.RegisterCommandHandler("reconnect", [&](const milvus::TelemetryCommand& command) {
        handler_entered = true;
        while (!allow_self_restart.load()) {
            std::this_thread::yield();
        }
        manager.Stop();
        manager.Start();
        self_restart_ready = manager.IsReady();
        return milvus::TelemetryCommandReply{command.command_id, true, "", ""};
    });

    manager.Start();
    for (int retry = 0; retry < 2000 && !handler_entered.load(); ++retry) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(handler_entered.load());
    ASSERT_TRUE(manager.IsReady());

    std::promise<void> external_stop_finished;
    auto external_stop_future = external_stop_finished.get_future();
    std::thread external_stopper([&]() {
        manager.Stop();
        external_stop_finished.set_value();
    });

    // ready=false is written before Stop() releases the manager mutex to join,
    // so observing it makes the intended race ordering deterministic.
    for (int retry = 0; retry < 2000 && manager.IsReady(); ++retry) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_FALSE(manager.IsReady());
    EXPECT_EQ(external_stop_future.wait_for(std::chrono::milliseconds(10)), std::future_status::timeout);

    allow_self_restart = true;
    EXPECT_EQ(external_stop_future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    external_stopper.join();
    server->Shutdown();

    EXPECT_FALSE(self_restart_ready.load());
    EXPECT_FALSE(manager.IsReady());
}

TEST(ClientTelemetryTest, DisabledTelemetryKeepsControlPlaneHeartbeatAlive) {
    ControlPlaneTelemetryService service;
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    ASSERT_NE(server, nullptr);

    milvus::TelemetryConfig config;
    config.heartbeat_interval_ms = 100;
    milvus::ClientTelemetryManager manager(config);
    manager.AttachChannel(grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()),
                          "", "", "", "", "");
    milvus::proto::milvus::SearchRequest request;
    manager.RecordOperation("Search", request, std::chrono::steady_clock::now(), true, "");

    manager.Start();
    for (int retry = 0; retry < 2000 && manager.Config().enabled; ++retry) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_FALSE(manager.Config().enabled);
    manager.Stop();
    manager.Start();
    ASSERT_TRUE(service.WaitForHeartbeats(2));
    ASSERT_FALSE(manager.Config().enabled);
    manager.RecordOperation("Search", request, std::chrono::steady_clock::now(), true, "");
    service.AllowEnable();
    ASSERT_TRUE(service.WaitForHeartbeats(3));
    manager.Stop();

    const auto requests = service.Requests();
    ASSERT_GE(requests.size(), 3U);
    EXPECT_EQ(requests[0].metrics_size(), 1);
    EXPECT_EQ(requests[1].metrics_size(), 0);
    ASSERT_EQ(requests[1].command_replies_size(), 1);
    EXPECT_EQ(requests[1].command_replies(0).command_id(), "disable");
    EXPECT_TRUE(requests[1].command_replies(0).success());
    const milvus::TelemetryCommand disable{"disable", "push_config", R"({"enabled":false})", 1, true, ""};
    EXPECT_EQ(requests[1].config_hash(), milvus::ClientTelemetryManager::CalculateConfigHash({disable}));

    EXPECT_EQ(requests[2].metrics_size(), 0);
    ASSERT_EQ(requests[2].command_replies_size(), 1);
    EXPECT_EQ(requests[2].command_replies(0).command_id(), "enable");
    EXPECT_TRUE(requests[2].command_replies(0).success());
    const milvus::TelemetryCommand enable{"enable", "push_config", R"({"enabled":true})", 2, true, ""};
    EXPECT_EQ(requests[2].config_hash(), milvus::ClientTelemetryManager::CalculateConfigHash({enable}));
    EXPECT_TRUE(manager.Config().enabled);

    server->Shutdown();
}

TEST(ClientTelemetryTest, InitialDisabledConfigDoesNotActivateControlPlane) {
    ControlPlaneTelemetryService service;
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    ASSERT_NE(server, nullptr);

    milvus::TelemetryConfig config;
    config.enabled = false;
    config.heartbeat_interval_ms = 1;
    milvus::ClientTelemetryManager manager(config);
    manager.AttachChannel(grpc::CreateChannel("127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()),
                          "", "", "", "", "");

    manager.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    manager.Stop();
    manager.Start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    manager.Stop();

    EXPECT_EQ(service.RequestCount(), 0U);
    server->Shutdown();
}

TEST(ClientTelemetryTest, UseDatabaseFromHeartbeatCommandReusesWorker) {
    LifecycleTelemetryService service("use_database");
    int port = 0;
    auto server = StartLifecycleServer(service, port);
    ASSERT_NE(server, nullptr);

    auto client = milvus::MilvusClient::Create();
    ASSERT_TRUE(client->Connect(TelemetryConnectParam(port)).IsOk());
    std::atomic<bool> command_finished{false};
    std::atomic<bool> use_database_succeeded{false};
    std::weak_ptr<milvus::MilvusClient> weak_client = client;
    client->GetTelemetry()->RegisterCommandHandler("use_database", [&](const milvus::TelemetryCommand& command) {
        auto current_client = weak_client.lock();
        use_database_succeeded = current_client != nullptr && current_client->UseDatabase("secondary").IsOk();
        command_finished = true;
        return milvus::TelemetryCommandReply{command.command_id, use_database_succeeded.load(), "", ""};
    });
    service.EnableCommands();

    for (int retry = 0; retry < 3000 && (!command_finished.load() || service.heartbeats.load() < 2 ||
                                         !service.saw_secondary_database.load());
         ++retry) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    EXPECT_TRUE(command_finished.load());
    EXPECT_TRUE(use_database_succeeded.load());
    EXPECT_GE(service.connects.load(), 2);
    EXPECT_GE(service.heartbeats.load(), 2);
    EXPECT_TRUE(service.saw_secondary_database.load());
    EXPECT_TRUE(client->Disconnect().IsOk());
    client.reset();
    server->Shutdown();
}

TEST(ClientTelemetryTest, AllLifecycleEntriesFailFastWhenConcurrentConnectWaitsForCommandLock) {
    LifecycleTelemetryService service("connect");
    int port = 0;
    auto server = StartLifecycleServer(service, port);
    ASSERT_NE(server, nullptr);

    const auto connect_param = TelemetryConnectParam(port);
    auto client = milvus::MilvusClient::Create();
    ASSERT_TRUE(client->Connect(connect_param).IsOk());

    std::atomic<bool> handler_started{false};
    auto handler_finished = std::make_shared<std::promise<std::vector<milvus::StatusCode>>>();
    auto handler_future = handler_finished->get_future();
    client->GetTelemetry()->RegisterCommandHandler(
        "connect", [&, handler_finished](const milvus::TelemetryCommand& command) {
            handler_started = true;
            if (!service.WaitForConnects(2)) {
                handler_finished->set_value({milvus::StatusCode::UNKNOWN_ERROR});
                return milvus::TelemetryCommandReply{command.command_id, false, "concurrent connect did not run", ""};
            }

            // The external Connect has completed its server handshake and owns lifecycle_mtx_,
            // but its AttachChannel is blocked on this handler's command_mutex. Every re-entrant
            // lifecycle entry must fail fast rather than wait and complete the lock cycle.
            std::vector<milvus::StatusCode> codes;
            codes.push_back(client->Connect(connect_param).Code());
            codes.push_back(client->UseDatabase("other").Code());
            codes.push_back(client->Disconnect().Code());
            codes.push_back(client->SetRpcDeadlineMs(1234).Code());
            codes.push_back(client->SetRetryParam(milvus::RetryParam{}).Code());
            bool all_busy = true;
            for (auto code : codes) {
                all_busy = all_busy && code == milvus::StatusCode::CLIENT_BUSY;
            }
            handler_finished->set_value(codes);
            return milvus::TelemetryCommandReply{command.command_id, all_busy, "", ""};
        });

    service.EnableCommands();
    for (int retry = 0; retry < 2000 && !handler_started.load(); ++retry) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(handler_started.load());

    milvus::Status concurrent_status;
    std::thread concurrent_connect([&]() { concurrent_status = client->Connect(connect_param); });
    const bool handler_failed_fast = handler_future.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    concurrent_connect.join();

    EXPECT_TRUE(handler_failed_fast);
    ASSERT_EQ(handler_future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    const auto codes = handler_future.get();
    ASSERT_EQ(codes.size(), 5U);
    for (auto code : codes) {
        EXPECT_EQ(code, milvus::StatusCode::CLIENT_BUSY);
    }
    EXPECT_TRUE(concurrent_status.IsOk()) << concurrent_status.Message();

    EXPECT_TRUE(client->Disconnect().IsOk());
    client.reset();
    server->Shutdown();
}

TEST(ClientTelemetryTest, FailedUseDatabasePreservesPublishedConnectionAndTelemetry) {
    LifecycleTelemetryService service("");
    int port = 0;
    auto server = StartLifecycleServer(service, port);
    ASSERT_NE(server, nullptr);

    auto connect_param = TelemetryConnectParam(port);
    connect_param.SetDbName("primary");
    auto client = milvus::MilvusClient::Create();
    ASSERT_TRUE(client->Connect(connect_param).IsOk());
    auto manager = client->GetTelemetry();
    ASSERT_NE(manager, nullptr);
    const auto client_id = manager->ClientId();

    service.FailNextConnect();
    const auto status = client->UseDatabase("secondary");
    EXPECT_EQ(status.Code(), milvus::StatusCode::SERVER_FAILED);
    ASSERT_EQ(client->GetTelemetry(), manager);
    EXPECT_EQ(manager->ClientId(), client_id);

    bool has_collection = false;
    EXPECT_TRUE(client->HasCollection("still-connected", has_collection).IsOk());
    EXPECT_EQ(service.has_collections.load(), 1);

    manager->ProcessCommands({{"config-after-failure", "get_config", "", 10, false, ""}});
    const auto replies = manager->PendingCommandReplies();
    ASSERT_FALSE(replies.empty());
    ASSERT_TRUE(replies.back().success) << replies.back().error_message;
    const auto user_config = nlohmann::json::parse(replies.back().payload).at("user_config");
    EXPECT_EQ(user_config.at("db_name"), "primary");

    EXPECT_TRUE(client->Disconnect().IsOk());
    client.reset();
    server->Shutdown();
}

TEST(ClientTelemetryTest, FailedConnectPreservesPublishedConnectionAndTelemetry) {
    LifecycleTelemetryService first_service("");
    LifecycleTelemetryService rejected_service("");
    int first_port = 0;
    int rejected_port = 0;
    auto first_server = StartLifecycleServer(first_service, first_port);
    auto rejected_server = StartLifecycleServer(rejected_service, rejected_port);
    ASSERT_NE(first_server, nullptr);
    ASSERT_NE(rejected_server, nullptr);

    auto first_param = TelemetryConnectParam(first_port);
    first_param.SetDbName("primary");
    auto client = milvus::MilvusClient::Create();
    ASSERT_TRUE(client->Connect(first_param).IsOk());
    auto manager = client->GetTelemetry();
    ASSERT_NE(manager, nullptr);
    const auto client_id = manager->ClientId();

    rejected_service.FailNextConnect();
    const auto status = client->Connect(TelemetryConnectParam(rejected_port));
    EXPECT_EQ(status.Code(), milvus::StatusCode::SERVER_FAILED);
    ASSERT_EQ(client->GetTelemetry(), manager);
    EXPECT_EQ(manager->ClientId(), client_id);

    bool has_collection = false;
    EXPECT_TRUE(client->HasCollection("still-on-first", has_collection).IsOk());
    EXPECT_EQ(first_service.has_collections.load(), 1);
    EXPECT_EQ(rejected_service.has_collections.load(), 0);

    manager->ProcessCommands({{"config-after-rejected-connect", "get_config", "", 10, false, ""}});
    const auto replies = manager->PendingCommandReplies();
    ASSERT_FALSE(replies.empty());
    ASSERT_TRUE(replies.back().success) << replies.back().error_message;
    const auto user_config = nlohmann::json::parse(replies.back().payload).at("user_config");
    EXPECT_EQ(user_config.at("address"), "127.0.0.1:" + std::to_string(first_port));
    EXPECT_EQ(user_config.at("db_name"), "primary");

    EXPECT_TRUE(client->Disconnect().IsOk());
    client.reset();
    first_server->Shutdown();
    rejected_server->Shutdown();
}

TEST(ClientTelemetryTest, NonStandardCommandExceptionReturnsFailureAndHeartbeatContinues) {
    LifecycleTelemetryService service("throw_non_standard");
    int port = 0;
    auto server = StartLifecycleServer(service, port);
    ASSERT_NE(server, nullptr);

    auto client = milvus::MilvusClient::Create();
    ASSERT_TRUE(client->Connect(TelemetryConnectParam(port)).IsOk());
    client->GetTelemetry()->RegisterCommandHandler(
        "throw_non_standard", [](const milvus::TelemetryCommand&) -> milvus::TelemetryCommandReply { throw 42; });
    service.EnableCommands();

    for (int retry = 0; retry < 3000 && (service.heartbeats.load() < 3 || !service.saw_failed_command_reply.load());
         ++retry) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_GE(service.heartbeats.load(), 3);
    EXPECT_TRUE(service.saw_failed_command_reply.load());

    EXPECT_TRUE(client->Disconnect().IsOk());
    client.reset();
    server->Shutdown();
}

TEST(ClientTelemetryTest, LastManagerReferenceCanBeDestroyedByHeartbeatCommand) {
    LifecycleTelemetryService service("disconnect_and_destroy");
    int port = 0;
    auto server = StartLifecycleServer(service, port);
    ASSERT_NE(server, nullptr);

    auto client = milvus::MilvusClient::Create();
    ASSERT_TRUE(client->Connect(TelemetryConnectParam(port)).IsOk());
    auto destroyed = std::make_shared<std::promise<void>>();
    auto destroyed_future = destroyed->get_future();
    auto marker = std::make_shared<DestructionSignal>(destroyed);
    client->GetTelemetry()->RegisterCommandHandler(
        "disconnect_and_destroy", [&client, marker](const milvus::TelemetryCommand& command) {
            auto status = client->Disconnect();
            client.reset();
            return milvus::TelemetryCommandReply{command.command_id, status.IsOk(), "", ""};
        });
    marker.reset();
    service.EnableCommands();

    ASSERT_EQ(destroyed_future.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_EQ(client, nullptr);
    server->Shutdown();
}

TEST(ClientTelemetryTest, V2LastManagerReferenceCanBeDestroyedByHeartbeatCommand) {
    LifecycleTelemetryService service("disconnect_and_destroy_v2");
    int port = 0;
    auto server = StartLifecycleServer(service, port);
    ASSERT_NE(server, nullptr);

    auto client = milvus::MilvusClientV2::Create();
    ASSERT_TRUE(client->Connect(TelemetryConnectParam(port)).IsOk());
    auto destroyed = std::make_shared<std::promise<void>>();
    auto destroyed_future = destroyed->get_future();
    auto marker = std::make_shared<DestructionSignal>(destroyed);
    client->GetTelemetry()->RegisterCommandHandler(
        "disconnect_and_destroy_v2", [&client, marker](const milvus::TelemetryCommand& command) {
            auto status = client->Disconnect();
            client.reset();
            return milvus::TelemetryCommandReply{command.command_id, status.IsOk(), "", ""};
        });
    marker.reset();
    service.EnableCommands();

    ASSERT_EQ(destroyed_future.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_EQ(client, nullptr);
    server->Shutdown();
}

TEST(ClientRequestContextTest, PropagatesOnlyValidTraceIdMetadataOnWire) {
    RequestMetadataService service;
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    ASSERT_NE(server, nullptr);

    auto client = milvus::MilvusClient::Create();
    auto connect_param = TelemetryConnectParam(port);
    milvus::TelemetryConfig telemetry_config;
    telemetry_config.enabled = false;
    connect_param.SetTelemetryConfig(telemetry_config);
    ASSERT_TRUE(client->Connect(connect_param).IsOk());

    bool has_collection = false;
    constexpr const char* valid_request_id = "4bf92f3577b34da6a3ce929d0e0e4736";
    {
        milvus::ScopedClientRequestId request_id(valid_request_id);
        EXPECT_TRUE(client->HasCollection("valid", has_collection).IsOk());
    }
    {
        milvus::ScopedClientRequestId request_id("");
        EXPECT_TRUE(client->HasCollection("empty", has_collection).IsOk());
    }
    {
        milvus::ScopedClientRequestId request_id("ABCDEF0123456789ABCDEF0123456789");
        EXPECT_TRUE(client->HasCollection("invalid", has_collection).IsOk());
    }

    const auto request_ids = service.RequestIds();
    ASSERT_EQ(request_ids.size(), 3U);
    EXPECT_TRUE(request_ids[0].first);
    EXPECT_EQ(request_ids[0].second, valid_request_id);
    EXPECT_FALSE(request_ids[1].first);
    EXPECT_FALSE(request_ids[2].first);

    EXPECT_TRUE(client->Disconnect().IsOk());
    client.reset();
    server->Shutdown();
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

TEST(ClientTelemetryTest, GlobalPhysicalHandoffAndUseDatabaseKeepLogicalIdentityAndState) {
    LifecycleTelemetryService first_service("");
    LifecycleTelemetryService second_service("");
    int first_port = 0;
    int second_port = 0;
    auto first_server = StartLifecycleServer(first_service, first_port);
    auto second_server = StartLifecycleServer(second_service, second_port);
    ASSERT_NE(first_server, nullptr);
    ASSERT_NE(second_server, nullptr);

    constexpr const char* logical_endpoint = "https://tenant.global-cluster.example.com";
    milvus::ConnectParam first_param("http://127.0.0.1:" + std::to_string(first_port));
    first_param.SetDbName("primary_db");
    milvus::TelemetryConfig telemetry_config;
    telemetry_config.enabled = false;
    first_param.SetTelemetryConfig(telemetry_config);

    auto first_connection = std::make_shared<milvus::MilvusConnection>();
    ASSERT_TRUE(first_connection->Connect(first_param, "", nullptr, logical_endpoint).IsOk());
    auto manager = first_connection->GetTelemetry();
    ASSERT_NE(manager, nullptr);
    const auto client_id = manager->ClientId();
    manager->ProcessCommands({
        {"config", "push_config", R"({"sampling_rate":0.25})", 1, true, ""},
        {"collections", "collection_metrics", R"({"enabled":true,"collections":["before_failover"]})", 2, false, ""},
    });
    const auto config_hash = manager->ConfigHash();
    const auto replies_before_failover = manager->PendingCommandReplies();
    ASSERT_EQ(replies_before_failover.size(), 2U);

    milvus::ConnectParam second_param = first_param;
    second_param.SetUri("http://127.0.0.1:" + std::to_string(second_port));
    auto second_connection = std::make_shared<milvus::MilvusConnection>();
    ASSERT_TRUE(second_connection->Connect(second_param, client_id, manager, logical_endpoint).IsOk());
    ASSERT_EQ(second_connection->GetTelemetry(), manager);
    EXPECT_EQ(manager->ClientId(), client_id);
    EXPECT_EQ(manager->ConfigHash(), config_hash);
    EXPECT_EQ(manager->LastCommandTimestamp(), 2);
    EXPECT_DOUBLE_EQ(manager->Config().sampling_rate, 0.25);
    EXPECT_EQ(manager->PendingCommandReplies().size(), replies_before_failover.size());
    EXPECT_TRUE(first_connection->Disconnect(false).IsOk());

    ASSERT_TRUE(second_connection->UseDatabase("secondary_db").IsOk());
    ASSERT_EQ(second_connection->GetTelemetry(), manager);
    EXPECT_EQ(manager->ClientId(), client_id);
    EXPECT_EQ(manager->ConfigHash(), config_hash);
    EXPECT_EQ(manager->LastCommandTimestamp(), 2);

    manager->ProcessCommands({{"config-after-use-db", "get_config", "", 3, false, ""}});
    const auto replies = manager->PendingCommandReplies();
    ASSERT_FALSE(replies.empty());
    ASSERT_TRUE(replies.back().success) << replies.back().error_message;
    const auto user_config = nlohmann::json::parse(replies.back().payload).at("user_config");
    EXPECT_EQ(user_config.at("address"), logical_endpoint);
    EXPECT_EQ(user_config.at("db_name"), "secondary_db");

    EXPECT_TRUE(second_connection->Disconnect().IsOk());
    first_server->Shutdown();
    second_server->Shutdown();
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

TEST(ClientTelemetryTest, RejectsInvalidRfc3339CalendarTimes) {
    milvus::TelemetryConfig config;
    config.enabled = false;
    milvus::ClientTelemetryManager manager(config);

    manager.ProcessCommands(
        {{"invalid-day", "show_latency_history",
          R"({"start_time":"2026-02-30T00:00:00Z","end_time":"2026-03-01T00:00:00Z","detail":false})", 1, false, ""},
         {"leap-second", "show_latency_history",
          R"({"start_time":"2026-02-28T23:59:60Z","end_time":"2026-03-01T00:00:00Z","detail":false})", 2, false, ""},
         {"missing-seconds", "show_latency_history",
          R"({"start_time":"2026-02-28T23:59Z","end_time":"2026-03-01T00:00:00Z","detail":false})", 3, false, ""}});

    const auto replies = manager.PendingCommandReplies();
    ASSERT_EQ(replies.size(), 3U);
    EXPECT_FALSE(replies[0].success);
    EXPECT_FALSE(replies[1].success);
    EXPECT_FALSE(replies[2].success);
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
