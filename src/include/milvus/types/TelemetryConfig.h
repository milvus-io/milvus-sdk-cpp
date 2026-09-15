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

#include <cstddef>
#include <cstdint>
#include <string>

#include "milvus/Export.h"

namespace milvus {

/**
 * @brief Client metrics, heartbeat, and server-pushed command configuration.
 */
struct MILVUS_SDK_API TelemetryConfig {
    /** @brief Whether telemetry reporting is enabled. */
    bool enabled{true};
    /** @brief Milliseconds between heartbeats, which is also the metrics window.
     *
     * Each heartbeat carries the operations since the last one. The coordinator answers a telemetry query from the
     * window before the newest, so what a caller reads is between one and two intervals old.
     */
    uint64_t heartbeat_interval_ms{10000};
    /** @brief Sampling rate of recorded operations (0.0 to 1.0). */
    double sampling_rate{1.0};
    /** @brief Maximum number of recorded errors retained. */
    size_t error_max_count{100};

    /** @brief Optional stable identity; a random UUID is used when empty. */
    std::string client_id;
};

}  // namespace milvus
