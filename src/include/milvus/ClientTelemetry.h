// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "milvus/Export.h"
#include "milvus/types/TelemetryConfig.h"

namespace grpc {
class Channel;
}

namespace google {
namespace protobuf {
class Message;
}
}  // namespace google

namespace milvus {

struct MILVUS_SDK_API TelemetryMetric {
    int64_t request_count{0};
    int64_t success_count{0};
    int64_t error_count{0};
    double avg_latency_ms{0};
    double p99_latency_ms{0};
    double max_latency_ms{0};
};

struct MILVUS_SDK_API TelemetryOperationMetrics {
    std::string operation;
    TelemetryMetric global;
    std::unordered_map<std::string, TelemetryMetric> collection_metrics;
};

struct MILVUS_SDK_API TelemetrySnapshot {
    int64_t timestamp{0};
    int64_t end_time{0};
    std::vector<TelemetryOperationMetrics> metrics;
};

struct MILVUS_SDK_API TelemetryError {
    int64_t timestamp{0};
    std::string operation;
    std::string error_message;
    std::string collection;
    std::string request_id;
};

struct MILVUS_SDK_API TelemetryCommand {
    std::string command_id;
    std::string command_type;
    std::string payload;
    int64_t create_time{0};
    bool persistent{false};
    std::string target_scope;
};

struct MILVUS_SDK_API TelemetryCommandReply {
    std::string command_id;
    bool success{false};
    std::string error_message;
    std::string payload;
};

/** Client-side metrics, heartbeat, command, and diagnostic manager. */
class MILVUS_SDK_API ClientTelemetryManager {
 public:
    using CommandHandler = std::function<TelemetryCommandReply(const TelemetryCommand&)>;

    explicit ClientTelemetryManager(const TelemetryConfig& config = TelemetryConfig{},
                                    const std::string& runtime_client_id = "");
    ~ClientTelemetryManager();

    ClientTelemetryManager(const ClientTelemetryManager&) = delete;
    ClientTelemetryManager&
    operator=(const ClientTelemetryManager&) = delete;

    void
    AttachChannel(const std::shared_ptr<grpc::Channel>& channel, const std::string& username,
                  const std::string& database, const std::string& uri, const std::string& sdk_version,
                  const std::string& connection_scope = "");

    void
    Start();

    void
    Stop();

    bool
    IsReady() const;

    bool
    IsSupported() const;

    const std::string&
    ClientId() const;

    std::string
    ConfigHash() const;

    int64_t
    LastCommandTimestamp() const;

    TelemetryConfig
    Config() const;

    /** Whether a reconnect can reuse this manager without changing user-supplied telemetry settings. */
    bool
    MatchesConnection(const TelemetryConfig& config, const std::string& connection_scope) const;

    std::string
    LastHeartbeatError() const;

    void
    RegisterCommandHandler(const std::string& command_type, CommandHandler handler);

    void
    RecordOperation(const std::string& operation, const google::protobuf::Message& request,
                    std::chrono::steady_clock::time_point started, bool success, const std::string& error_message,
                    const std::string& request_id = "");

    void
    RecordOperation(const std::string& operation, const std::string& collection,
                    std::chrono::steady_clock::time_point started, bool success, const std::string& error_message,
                    const std::string& request_id = "");

    std::vector<TelemetryError>
    RecentErrors(size_t max_count = 100) const;

    std::vector<TelemetrySnapshot>
    MetricsSnapshots() const;

    std::vector<TelemetryCommandReply>
    PendingCommandReplies() const;

    void
    ProcessCommands(const std::vector<TelemetryCommand>& commands);

    static std::string
    CalculateConfigHash(const std::vector<TelemetryCommand>& commands);

 private:
    friend class MilvusClient;
    friend class MilvusClientV2;

    bool
    isWorkerThread() const;

    class Impl;
    std::shared_ptr<Impl> impl_;
};

using ClientTelemetryManagerPtr = std::shared_ptr<ClientTelemetryManager>;

}  // namespace milvus
