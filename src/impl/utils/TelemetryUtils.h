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
#include <utility>

#include "ConnectionHandler.h"
#include "milvus/ClientRequestContext.h"

namespace milvus {

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
