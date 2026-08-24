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

/** Per-thread request ID propagated as the client_request_id gRPC metadata. */
class MILVUS_SDK_API ClientRequestContext {
 public:
    static void
    Set(const std::string& request_id);

    static const std::string&
    Get();

    static void
    Clear();

    /** Returns a lowercase 32-character OpenTelemetry-compatible trace ID. */
    static std::string
    NewRequestId();

    /** Returns true for a lowercase, non-zero, 32-character OpenTelemetry trace ID. */
    static bool
    IsValid(const std::string& request_id);
};

/** Restores the previous thread-local request ID when it leaves scope. */
class MILVUS_SDK_API ScopedClientRequestId {
 public:
    explicit ScopedClientRequestId(const std::string& request_id);
    ~ScopedClientRequestId();

    ScopedClientRequestId(const ScopedClientRequestId&) = delete;
    ScopedClientRequestId&
    operator=(const ScopedClientRequestId&) = delete;

 private:
    std::string previous_;
};

}  // namespace milvus
