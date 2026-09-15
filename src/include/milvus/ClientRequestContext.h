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

#include <string>

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Per-thread request ID propagated as the client_request_id gRPC metadata.
 */
class MILVUS_SDK_API ClientRequestContext {
 public:
    /**
     * @brief Set the request ID of the current thread.
     *
     * @param [in] request_id request ID to propagate in outgoing RPC metadata.
     */
    static void
    Set(const std::string& request_id);

    /**
     * @brief Get the request ID of the current thread.
     * @return the request id.
     */
    static const std::string&
    Get();

    /**
     * @brief Clear the request ID of the current thread.
     */
    static void
    Clear();

    /**
     * @brief Generate a lowercase 32-character OpenTelemetry-compatible trace ID.
     * @return the new request ID.
     */
    static std::string
    NewRequestId();

    /**
     * @brief Check whether a string is a valid lowercase, non-zero 32-character OpenTelemetry trace ID.
     *
     * @param [in] request_id request ID to validate.
     * @return true when the request ID is a valid trace ID.
     */
    static bool
    IsValid(const std::string& request_id);
};

/**
 * @brief RAII guard that restores the previous thread-local request ID when it leaves scope.
 */
class MILVUS_SDK_API ScopedClientRequestId {
 public:
    /**
     * @brief Set a request ID for the current thread and remember the previous one.
     *
     * @param [in] request_id request ID to install for the current scope.
     */
    explicit ScopedClientRequestId(const std::string& request_id);

    /**
     * @brief Restore the previous thread-local request ID.
     */
    ~ScopedClientRequestId();

    ScopedClientRequestId(const ScopedClientRequestId&) = delete;
    ScopedClientRequestId&
    operator=(const ScopedClientRequestId&) = delete;

 private:
    std::string previous_;
};

}  // namespace milvus
