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

/**
 * @brief Aggregated metrics for one operation within a telemetry window.
 */
struct MILVUS_SDK_API TelemetryMetric {
    /** @brief number of RPC attempts issued. */
    int64_t request_count{0};
    /** @brief number of RPC attempts that succeeded. */
    int64_t success_count{0};
    /** @brief number of RPC attempts that failed. */
    int64_t error_count{0};
    /** @brief average latency in milliseconds. */
    double avg_latency_ms{0};
    /** @brief p99 latency in milliseconds. */
    double p99_latency_ms{0};
    /** @brief maximum latency in milliseconds. */
    double max_latency_ms{0};
};

/**
 * @brief Metrics of one operation type, aggregated globally and per collection.
 */
struct MILVUS_SDK_API TelemetryOperationMetrics {
    /** @brief operation name, e.g. "Insert" or "Search". */
    std::string operation;
    /** @brief metrics aggregated across all collections. */
    TelemetryMetric global;
    /** @brief metrics aggregated per collection name. */
    std::unordered_map<std::string, TelemetryMetric> collection_metrics;
};

/**
 * @brief A telemetry window carrying the operation metrics collected since the previous heartbeat.
 */
struct MILVUS_SDK_API TelemetrySnapshot {
    /** @brief start timestamp of the window. */
    int64_t timestamp{0};
    /** @brief end timestamp of the window. */
    int64_t end_time{0};
    /** @brief per-operation metrics within this window. */
    std::vector<TelemetryOperationMetrics> metrics;
};

/**
 * @brief One recorded RPC error reported by the telemetry manager.
 */
struct MILVUS_SDK_API TelemetryError {
    /** @brief timestamp when the error occurred. */
    int64_t timestamp{0};
    /** @brief operation name that produced the error. */
    std::string operation;
    /** @brief error message. */
    std::string error_message;
    /** @brief target collection, empty when not applicable. */
    std::string collection;
    /** @brief client request id of the failing call, empty when not available. */
    std::string request_id;
};

/**
 * @brief A server-pushed telemetry command (for example a config update or a metrics toggle).
 */
struct MILVUS_SDK_API TelemetryCommand {
    /** @brief unique command identifier. */
    std::string command_id;
    /** @brief command type, e.g. "push_config" or "get_config". */
    std::string command_type;
    /** @brief command payload as a JSON string. */
    std::string payload;
    /** @brief creation timestamp of the command. */
    int64_t create_time{0};
    /** @brief whether the command must survive a client reconnect. */
    bool persistent{false};
    /** @brief scope the command targets, e.g. "config" or "collections". */
    std::string target_scope;
};

/**
 * @brief Reply to a telemetry command.
 */
struct MILVUS_SDK_API TelemetryCommandReply {
    /** @brief identifier of the command being replied to. */
    std::string command_id;
    /** @brief whether the command was handled successfully. */
    bool success{false};
    /** @brief error message when the command failed. */
    std::string error_message;
    /** @brief reply payload as a JSON string. */
    std::string payload;
};

/**
 * @brief Client-side metrics, heartbeat, command, and diagnostic manager.
 */
class MILVUS_SDK_API ClientTelemetryManager {
 public:
    using CommandHandler = std::function<TelemetryCommandReply(const TelemetryCommand&)>;

    /**
     * @brief Create a telemetry manager.
     *
     * @param [in] config telemetry configuration; telemetry is disabled when config.enabled is false.
     * @param [in] runtime_client_id explicit client identity; a random UUID is used when empty.
     */
    explicit ClientTelemetryManager(const TelemetryConfig& config = TelemetryConfig{},
                                    const std::string& runtime_client_id = "");

    /**
     * @brief Destroy the telemetry manager and stop its worker thread.
     */
    ~ClientTelemetryManager();

    ClientTelemetryManager(const ClientTelemetryManager&) = delete;
    ClientTelemetryManager&
    operator=(const ClientTelemetryManager&) = delete;

    /**
     * @brief Attach the manager to a new connection.
     *
     * @param [in] channel gRPC channel of the connection.
     * @param [in] username authenticated user name.
     * @param [in] database current database name.
     * @param [in] uri connection endpoint.
     * @param [in] sdk_version SDK version string reported in heartbeats.
     * @param [in] connection_scope connection-scoped identity used to reuse the manager across reconnects.
     */
    void
    AttachChannel(const std::shared_ptr<grpc::Channel>& channel, const std::string& username,
                  const std::string& database, const std::string& uri, const std::string& sdk_version,
                  const std::string& connection_scope = "");

    /**
     * @brief Start the heartbeat/metrics worker thread.
     */
    void
    Start();

    /**
     * @brief Stop the heartbeat/metrics worker thread.
     */
    void
    Stop();

    /**
     * @brief Whether the manager is attached to a channel and started.
     * @return true if the manager is ready.
     */
    bool
    IsReady() const;

    /**
     * @brief Whether the connected server reports telemetry support (no consecutive unsupported responses).
     * @return true if telemetry is supported.
     */
    bool
    IsSupported() const;

    /**
     * @brief The client identity used in heartbeats and diagnostics.
     * @return the client ID.
     */
    const std::string&
    ClientId() const;

    /**
     * @brief Hash of the effective telemetry configuration, stable across failover.
     * @return the config hash.
     */
    std::string
    ConfigHash() const;

    /**
     * @brief Timestamp of the most recently processed server command, or 0.
     * @return the last command timestamp.
     */
    int64_t
    LastCommandTimestamp() const;

    /**
     * @brief The current effective telemetry configuration.
     * @return the config.
     */
    TelemetryConfig
    Config() const;

    /**
     * @brief Whether a reconnect can reuse this manager without changing user-supplied telemetry settings.
     *
     * @param [in] config telemetry configuration of the new connection.
     * @param [in] connection_scope scope of the new connection.
     */
    bool
    MatchesConnection(const TelemetryConfig& config, const std::string& connection_scope) const;

    /**
     * @brief Error message of the last heartbeat RPC, empty when none has failed.
     * @return the last heartbeat error.
     */
    std::string
    LastHeartbeatError() const;

    /**
     * @brief Register a handler for a server command type.
     *
     * @param [in] command_type command type handled by the callback.
     * @param [in] handler callback invoked for matching commands.
     */
    void
    RegisterCommandHandler(const std::string& command_type, CommandHandler handler);

    /**
     * @brief Record metrics for an RPC operation, deriving the target collection from the request message.
     *
     * @param [in] operation operation name; only DML/DQL operations are recorded.
     * @param [in] request RPC request message used to derive the collection name.
     * @param [in] started start time of the RPC call.
     * @param [in] success whether the RPC succeeded.
     * @param [in] error_message error message when the RPC failed.
     * @param [in] request_id client request id of the call.
     */
    void
    RecordOperation(const std::string& operation, const google::protobuf::Message& request,
                    std::chrono::steady_clock::time_point started, bool success, const std::string& error_message,
                    const std::string& request_id = "");

    /**
     * @brief Record metrics for an RPC operation on an explicit collection.
     *
     * @param [in] operation operation name; only DML/DQL operations are recorded.
     * @param [in] collection target collection name, empty for global operations.
     * @param [in] started start time of the RPC call.
     * @param [in] success whether the RPC succeeded.
     * @param [in] error_message error message when the RPC failed.
     * @param [in] request_id client request id of the call.
     */
    void
    RecordOperation(const std::string& operation, const std::string& collection,
                    std::chrono::steady_clock::time_point started, bool success, const std::string& error_message,
                    const std::string& request_id = "");

    /**
     * @brief Get the most recent recorded errors.
     *
     * @param [in] max_count maximum number of errors to return.
     * @return recent error records, oldest first.
     */
    std::vector<TelemetryError>
    RecentErrors(size_t max_count = 100) const;

    /**
     * @brief Get the telemetry windows available for the current configuration window.
     * @return the metrics snapshots.
     */
    std::vector<TelemetrySnapshot>
    MetricsSnapshots() const;

    /**
     * @brief Get command replies not yet delivered back to the server.
     * @return the pending command replies.
     */
    std::vector<TelemetryCommandReply>
    PendingCommandReplies() const;

    /**
     * @brief Process server-pushed commands and generate replies.
     *
     * @param [in] commands commands to process.
     */
    void
    ProcessCommands(const std::vector<TelemetryCommand>& commands);

    /**
     * @brief Compute the configuration hash a server would derive from the given commands.
     *
     * @param [in] commands commands whose effective config is hashed.
     * @return configuration hash string.
     */
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
