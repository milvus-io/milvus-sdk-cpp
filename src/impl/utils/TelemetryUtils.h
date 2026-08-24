// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <chrono>
#include <string>
#include <thread>
#include <utility>

#include "ConnectionHandler.h"
#include "milvus/ClientRequestContext.h"

namespace milvus {

template <typename Client>
void
DeleteClientWithTelemetryWorkerSafety(Client* client, ClientTelemetryManagerPtr telemetry,
                                      bool called_from_telemetry_worker) noexcept {
    if (!called_from_telemetry_worker) {
        delete client;
        return;
    }

    // The global topology refresher may be waiting for the current command handler before
    // it can hand off the telemetry channel. Stop/join the worker on another thread, then
    // delete the client after the handler has returned and released the command lock.
    try {
        std::thread([client, telemetry = std::move(telemetry)]() {
            telemetry->Stop();
            delete client;
        }).detach();
    } catch (...) {
        // A deleter must not throw. If the one-shot cleanup thread cannot be created,
        // intentionally retain the client rather than terminate or re-enter the known
        // refresher/command join cycle.
    }
}

template <typename Callable>
Status
InvokeWithTelemetry(ConnectionHandler& connection, const std::string& operation, const std::string& collection,
                    Callable&& callable) {
    auto started = std::chrono::steady_clock::now();
    auto telemetry = connection.GetTelemetry();
    auto status = std::forward<Callable>(callable)();
    if (telemetry != nullptr) {
        const auto& request_id = ClientRequestContext::Get();
        telemetry->RecordOperation(operation, collection, started, status.IsOk(),
                                   status.IsOk() ? std::string{} : status.Message(), request_id);
    }
    return status;
}

}  // namespace milvus
