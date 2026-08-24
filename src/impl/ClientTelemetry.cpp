// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

#include "milvus/ClientTelemetry.h"

#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>

#include <milvus/thirdparty/nlohmann/json.hpp>

#include "common.pb.h"
#include "milvus.grpc.pb.h"
#include "milvus.pb.h"
#include "milvus/ClientRequestContext.h"

namespace milvus {
namespace {

constexpr size_t kSampleBufferSize = 1000;
constexpr size_t kStoredQuantileSampleCount = 128;
constexpr int64_t kMaxHistoryRangeMs = 60 * 60 * 1000;
// A one-second heartbeat produces at most 3,601 boundary-inclusive snapshots in
// one hour. Keep the complete protocol window plus margin while retaining a hard
// memory bound for shorter server-pushed intervals. Each operation/window stores
// at most 128 sorted, evenly spaced quantile samples including its minimum and
// maximum. Across seven supported operations this caps latency history at
// 3,670,016 doubles (about 28 MiB), plus snapshot/collection metric metadata.
constexpr size_t kSnapshotLimit = 4096;
// Fixed-point unit for accumulating a fractional sampling rate. A rate becomes an integer
// step of this many units, so the smallest rate that still samples is 1e-9 -- far below
// anything an operator would set, which is the point: a configured rate must never round
// down to "off".
constexpr uint64_t kSamplingScale = 1000000000ULL;
constexpr size_t kMaxReplyBytes = 1024 * 1024;
constexpr uint64_t kMaxUnsupportedBackoffMs = 30 * 60 * 1000;

TelemetryConfig
NormalizedTelemetryConfig(TelemetryConfig config) {
    if (config.heartbeat_interval_ms == 0) {
        config.heartbeat_interval_ms = 10000;
    }
    config.sampling_rate = std::max(0.0, std::min(1.0, config.sampling_rate));
    if (config.error_max_count == 0) {
        config.error_max_count = 100;
    }
    return config;
}

bool
SameTelemetryConfig(const TelemetryConfig& left, const TelemetryConfig& right) {
    return left.enabled == right.enabled && left.heartbeat_interval_ms == right.heartbeat_interval_ms &&
           left.sampling_rate == right.sampling_rate && left.error_max_count == right.error_max_count &&
           left.client_id == right.client_id;
}

int64_t
NowMillis() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string
RandomUuid() {
    std::random_device device;
    std::mt19937_64 generator(device());
    std::uniform_int_distribution<uint64_t> distribution;
    auto high = distribution(generator);
    auto low = distribution(generator);
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(8) << static_cast<uint32_t>(high >> 32) << "-" << std::setw(4)
           << static_cast<uint16_t>(high >> 16) << "-" << std::setw(4) << static_cast<uint16_t>(high) << "-"
           << std::setw(4) << static_cast<uint16_t>(low >> 48) << "-" << std::setw(12) << (low & 0x0000FFFFFFFFFFFFULL);
    return stream.str();
}

std::string
LocalTimeString() {
    auto now = std::chrono::system_clock::now();
    auto value = std::chrono::system_clock::to_time_t(now);
    std::tm time{};
#ifdef _WIN32
    gmtime_s(&time, &value);
#else
    gmtime_r(&value, &time);
#endif
    std::ostringstream stream;
    stream << std::put_time(&time, "%Y-%m-%dT%H:%M:%SZ");
    return stream.str();
}

std::string
HostName() {
#ifdef _WIN32
    const char* value = std::getenv("COMPUTERNAME");
    return value == nullptr ? "Unknown" : value;
#else
    std::array<char, 256> buffer{};
    if (gethostname(buffer.data(), buffer.size()) != 0) {
        return "Unknown";
    }
    buffer.back() = '\0';
    return {buffer.data()};
#endif
}

std::string
CollectionName(const google::protobuf::Message& request) {
    const auto* field = request.GetDescriptor()->FindFieldByName("collection_name");
    if (field == nullptr || field->cpp_type() != google::protobuf::FieldDescriptor::CPPTYPE_STRING) {
        return "";
    }
    return request.GetReflection()->GetString(request, field);
}

uint32_t
RotateRight(uint32_t value, uint32_t count) {
    return (value >> count) | (value << (32 - count));
}

// Small self-contained SHA-256 implementation keeps the SDK independent from a specific TLS provider.
class Sha256 {
 public:
    Sha256() : state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19} {
    }

    void
    Update(const std::string& value) {
        update(reinterpret_cast<const uint8_t*>(value.data()), value.size());
    }

    std::string
    Finish() {
        uint64_t bit_length = total_size_ * 8;
        buffer_[buffer_size_++] = 0x80;
        if (buffer_size_ > 56) {
            while (buffer_size_ < 64) {
                buffer_[buffer_size_++] = 0;
            }
            transform(buffer_.data());
            buffer_size_ = 0;
        }
        while (buffer_size_ < 56) {
            buffer_[buffer_size_++] = 0;
        }
        for (int index = 7; index >= 0; --index) {
            buffer_[buffer_size_++] = static_cast<uint8_t>(bit_length >> (index * 8));
        }
        transform(buffer_.data());

        std::ostringstream stream;
        stream << std::hex << std::setfill('0');
        for (auto value : state_) {
            stream << std::setw(8) << value;
        }
        return stream.str();
    }

 private:
    void
    update(const uint8_t* data, size_t size) {
        total_size_ += size;
        for (size_t index = 0; index < size; ++index) {
            buffer_[buffer_size_++] = data[index];
            if (buffer_size_ == 64) {
                transform(buffer_.data());
                buffer_size_ = 0;
            }
        }
    }

    void
    transform(const uint8_t* block) {
        static const std::array<uint32_t, 64> constants = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        std::array<uint32_t, 64> schedule{};
        for (size_t index = 0; index < 16; ++index) {
            schedule[index] =
                (static_cast<uint32_t>(block[index * 4]) << 24) | (static_cast<uint32_t>(block[index * 4 + 1]) << 16) |
                (static_cast<uint32_t>(block[index * 4 + 2]) << 8) | static_cast<uint32_t>(block[index * 4 + 3]);
        }
        for (size_t index = 16; index < 64; ++index) {
            uint32_t first = RotateRight(schedule[index - 15], 7) ^ RotateRight(schedule[index - 15], 18) ^
                             (schedule[index - 15] >> 3);
            uint32_t second = RotateRight(schedule[index - 2], 17) ^ RotateRight(schedule[index - 2], 19) ^
                              (schedule[index - 2] >> 10);
            schedule[index] = schedule[index - 16] + first + schedule[index - 7] + second;
        }
        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (size_t index = 0; index < 64; ++index) {
            uint32_t sum1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
            uint32_t choice = (e & f) ^ ((~e) & g);
            uint32_t temp1 = h + sum1 + choice + constants[index] + schedule[index];
            uint32_t sum0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
            uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<uint32_t, 8> state_;
    std::array<uint8_t, 64> buffer_{};
    size_t buffer_size_{0};
    uint64_t total_size_{0};
};

int64_t
DaysFromCivil(int year, unsigned month, unsigned day) {
    year -= month <= 2;
    const int era = (year >= 0 ? year : year - 399) / 400;
    const auto year_of_era = static_cast<unsigned>(year - era * 400);
    const unsigned adjusted_month = month > 2 ? month - 3 : month + 9;
    const unsigned day_of_year = (153 * adjusted_month + 2) / 5 + day - 1;
    const unsigned day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    return static_cast<int64_t>(era) * 146097 + static_cast<int64_t>(day_of_era) - 719468;
}

int64_t
ParseRfc3339Millis(const std::string& value) {
    if (value.size() < 20 || value[4] != '-' || value[7] != '-' || value[10] != 'T' || value[13] != ':' ||
        value[16] != ':') {
        throw std::invalid_argument("invalid RFC3339 timestamp");
    }
    for (size_t index = 0; index < 19; ++index) {
        if (index == 4 || index == 7 || index == 10 || index == 13 || index == 16) {
            continue;
        }
        if (!std::isdigit(static_cast<unsigned char>(value[index]))) {
            throw std::invalid_argument("invalid RFC3339 timestamp");
        }
    }
    const int year = std::stoi(value.substr(0, 4));
    const int month = std::stoi(value.substr(5, 2));
    const int day = std::stoi(value.substr(8, 2));
    const int hour = std::stoi(value.substr(11, 2));
    const int minute = std::stoi(value.substr(14, 2));
    const int second = std::stoi(value.substr(17, 2));
    const bool leap_year = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    constexpr std::array<int, 12> kDaysPerMonth = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (month < 1 || month > 12 || day < 1 ||
        day > kDaysPerMonth[static_cast<size_t>(month - 1)] + (month == 2 && leap_year ? 1 : 0) || hour > 23 ||
        minute > 59 || second > 59) {
        throw std::invalid_argument("invalid RFC3339 timestamp");
    }
    size_t position = 19;
    int64_t milliseconds = 0;
    if (position < value.size() && value[position] == '.') {
        size_t end = position + 1;
        while (end < value.size() && std::isdigit(static_cast<unsigned char>(value[end]))) {
            ++end;
        }
        if (end == position + 1) {
            throw std::invalid_argument("invalid RFC3339 timestamp");
        }
        auto fraction = value.substr(position + 1, end - position - 1);
        while (fraction.size() < 3) {
            fraction.push_back('0');
        }
        milliseconds = std::stoll(fraction.substr(0, 3));
        position = end;
    }
    int offset_seconds = 0;
    if (position >= value.size()) {
        throw std::invalid_argument("invalid RFC3339 timezone");
    }
    if (value[position] == 'Z') {
        ++position;
    } else {
        if (position + 6 != value.size() || (value[position] != '+' && value[position] != '-') ||
            value[position + 3] != ':') {
            throw std::invalid_argument("invalid RFC3339 timezone");
        }
        for (auto index : {position + 1, position + 2, position + 4, position + 5}) {
            if (!std::isdigit(static_cast<unsigned char>(value[index]))) {
                throw std::invalid_argument("invalid RFC3339 timezone");
            }
        }
        int sign = value[position] == '+' ? 1 : -1;
        int hours = std::stoi(value.substr(position + 1, 2));
        int minutes = std::stoi(value.substr(position + 4, 2));
        if (hours > 23 || minutes > 59) {
            throw std::invalid_argument("invalid RFC3339 timezone");
        }
        offset_seconds = sign * (hours * 3600 + minutes * 60);
        position += 6;
    }
    if (position != value.size()) {
        throw std::invalid_argument("invalid RFC3339 timestamp");
    }
    const auto seconds = DaysFromCivil(year, static_cast<unsigned>(month), static_cast<unsigned>(day)) * 86400 +
                         hour * 3600 + minute * 60 + second - offset_seconds;
    return seconds * 1000 + milliseconds;
}

struct MetricBucket {
    int64_t requests{0};
    int64_t successes{0};
    int64_t errors{0};
    double total_latency_ms{0};
    double max_latency_ms{0};
    std::deque<double> samples;

    void
    Record(double latency_ms, bool success) {
        ++requests;
        success ? ++successes : ++errors;
        total_latency_ms += latency_ms;
        max_latency_ms = std::max(max_latency_ms, latency_ms);
        samples.push_back(latency_ms);
        if (samples.size() > kSampleBufferSize) {
            samples.pop_front();
        }
    }

    TelemetryMetric
    Snapshot(std::vector<double>* quantile_samples = nullptr) const {
        std::vector<double> sorted(samples.begin(), samples.end());
        std::sort(sorted.begin(), sorted.end());
        if (quantile_samples != nullptr) {
            quantile_samples->clear();
            const auto count = std::min(sorted.size(), kStoredQuantileSampleCount);
            quantile_samples->reserve(count);
            if (count == 1) {
                quantile_samples->push_back(sorted.front());
            } else if (count > 1) {
                // Index 0 and count-1 map exactly to the minimum and maximum. The
                // intermediate integer indices are evenly spaced over the sorted
                // reservoir and keep the output compact and already merge-ready.
                for (size_t index = 0; index < count; ++index) {
                    const auto source_index = index * (sorted.size() - 1) / (count - 1);
                    quantile_samples->push_back(sorted[source_index]);
                }
            }
        }
        auto index = sorted.empty() ? 0 : std::min(sorted.size() - 1, static_cast<size_t>(sorted.size() * 0.99));
        return {requests,
                successes,
                errors,
                requests == 0 ? 0 : total_latency_ms / requests,
                sorted.empty() ? 0 : sorted[index],
                max_latency_ms};
    }
};

struct OperationCollector {
    MetricBucket global;
    std::unordered_map<std::string, MetricBucket> collections;
};

struct StoredSnapshot {
    TelemetrySnapshot snapshot;
    std::unordered_map<std::string, std::vector<double>> global_samples;
};

proto::common::Metrics
ToProtoMetric(const TelemetryMetric& metric) {
    proto::common::Metrics result;
    result.set_request_count(metric.request_count);
    result.set_success_count(metric.success_count);
    result.set_error_count(metric.error_count);
    result.set_avg_latency_ms(metric.avg_latency_ms);
    result.set_p99_latency_ms(metric.p99_latency_ms);
    result.set_max_latency_ms(metric.max_latency_ms);
    return result;
}

nlohmann::json
MetricJson(const TelemetryMetric& metric) {
    return {{"request_count", metric.request_count},   {"success_count", metric.success_count},
            {"error_count", metric.error_count},       {"avg_latency_ms", metric.avg_latency_ms},
            {"p99_latency_ms", metric.p99_latency_ms}, {"max_latency_ms", metric.max_latency_ms}};
}

TelemetryCommandReply
SuccessReply(const std::string& command_id, std::string payload = "") {
    return {command_id, true, "", std::move(payload)};
}

TelemetryCommandReply
FailedReply(const std::string& command_id, const std::string& error) {
    return {command_id, false, error, ""};
}

}  // namespace

class ClientTelemetryManager::Impl : public std::enable_shared_from_this<ClientTelemetryManager::Impl> {
 public:
    Impl(const TelemetryConfig& value, const std::string& runtime_client_id)
        : config(NormalizedTelemetryConfig(value)),
          connection_config(config),
          stable_client_id(!value.client_id.empty()),
          client_id(stable_client_id ? value.client_id
                                     : (runtime_client_id.empty() ? RandomUuid() : runtime_client_id)) {
        RegisterDefaultHandlers();
    }

    ~Impl() {
        Stop();
    }

    void
    AttachChannel(const std::shared_ptr<grpc::Channel>& channel, const std::string& user, const std::string& db,
                  const std::string& endpoint, const std::string& version, const std::string& scope) {
        // Construct every potentially-throwing part before taking either lock. Once locked, only
        // noexcept shared_ptr/string moves and scalar updates remain, so a failed candidate cannot
        // partially replace the live telemetry transport or its identity fields.
        auto stub_holder = proto::milvus::ClientTelemetryService::NewStub(channel);
        auto new_stub = std::shared_ptr<proto::milvus::ClientTelemetryService::Stub>(std::move(stub_holder));
        std::string new_username = user;
        std::string new_database = db;
        std::string new_uri = endpoint;
        std::string new_sdk_version = version;
        std::string new_connection_scope = scope;

        std::lock_guard<std::recursive_mutex> command_lock(command_mutex);
        std::lock_guard<std::mutex> lock(mutex);
        stub = std::move(new_stub);
        ++channel_generation;
        username = std::move(new_username);
        database = std::move(new_database);
        uri = std::move(new_uri);
        sdk_version = std::move(new_sdk_version);
        connection_scope = std::move(new_connection_scope);
    }

    void
    Start() {
        std::unique_lock<std::mutex> lock(mutex);
        if (ready) {
            return;
        }
        const bool called_from_worker = worker_running && worker_id == std::this_thread::get_id();
        if (called_from_worker) {
            // Stop()/Start() is used by reconnect handlers. Reuse this worker only
            // when no external caller has claimed it for joining. An external stop
            // always wins over a self-restart that races with it.
            if (join_in_progress || external_stop_requested) {
                return;
            }
            ready = true;
            control_plane_activated = control_plane_activated || config.enabled;
            if (!control_plane_activated) {
                stopped = true;
                return;
            }
            stopped = false;
            return;
        }

        condition.wait(lock, [this]() { return !join_in_progress; });
        if (ready) {
            return;
        }
        if (worker.joinable()) {
            join_in_progress = true;
            auto previous_worker = std::move(worker);
            lock.unlock();
            previous_worker.join();
            lock.lock();
            worker_id = {};
            worker_running = false;
            join_in_progress = false;
            condition.notify_all();
            if (ready) {
                return;
            }
        }
        external_stop_requested = false;
        ready = true;
        // Initial enabled=false is an explicit opt-out and creates no control-plane traffic.
        // Once activated, keep the control plane sticky across dynamic disable and
        // Stop/Start reconnects so the server can later re-enable telemetry.
        control_plane_activated = control_plane_activated || config.enabled;
        if (!control_plane_activated) {
            stopped = true;
            return;
        }
        stopped = false;
        worker_running = true;
        try {
            auto self = shared_from_this();
            worker = std::thread([self = std::move(self)]() { self->HeartbeatLoop(); });
            worker_id = worker.get_id();
        } catch (...) {
            // Telemetry startup is best-effort. Restore a clean, retryable stopped state instead
            // of leaking an exception through Connect() or leaving worker_running without a thread.
            ready = false;
            stopped = true;
            worker_running = false;
            worker_id = {};
            condition.notify_all();
        }
    }

    void
    Stop() {
        std::unique_lock<std::mutex> lock(mutex);
        stopped = true;
        ready = false;
        condition.notify_all();

        const bool called_from_worker = worker_running && worker_id == std::this_thread::get_id();
        if (called_from_worker) {
            return;
        }

        external_stop_requested = true;
        condition.wait(lock, [this]() { return !join_in_progress; });
        if (!worker.joinable()) {
            worker_id = {};
            worker_running = false;
            return;
        }

        // Move the thread object while holding the state mutex. This gives
        // exactly one external caller ownership of join(); other Stop()/Start()
        // calls wait on join_in_progress instead of touching the same std::thread.
        join_in_progress = true;
        auto current_worker = std::move(worker);
        lock.unlock();
        current_worker.join();
        lock.lock();
        worker_id = {};
        worker_running = false;
        join_in_progress = false;
        condition.notify_all();
    }

    void
    HeartbeatLoop() {
        while (true) {
            try {
                CreateSnapshot();
                SendHeartbeat();
            } catch (...) {
                // No telemetry collection, serialization, transport, or extension failure may
                // escape a std::thread entry and terminate the process. Keep the control plane
                // alive; the next iteration can retry on the same or a reattached transport.
                try {
                    std::lock_guard<std::mutex> lock(mutex);
                    last_heartbeat_error = "unexpected client telemetry heartbeat failure";
                } catch (...) {
                    // Even recording a best-effort diagnostic can fail under memory pressure.
                }
            }
            std::unique_lock<std::mutex> lock(mutex);
            if (stopped) {
                break;
            }
            uint64_t delay = config.heartbeat_interval_ms;
            if (unsupported_streak > 0) {
                uint64_t backed_off = std::min(delay, kMaxUnsupportedBackoffMs);
                for (int index = 0; index < unsupported_streak && backed_off < kMaxUnsupportedBackoffMs; ++index) {
                    backed_off = backed_off > kMaxUnsupportedBackoffMs / 2 ? kMaxUnsupportedBackoffMs : backed_off * 2;
                }
                delay = std::max(delay, backed_off);
            }
            condition.wait_for(lock, std::chrono::milliseconds(delay), [this]() { return stopped; });
            if (stopped) {
                break;
            }
        }
        std::lock_guard<std::mutex> lock(mutex);
        worker_running = false;
        // If Stop() was called by a command handler, no external thread owns
        // join(). Detach only after the loop has finished using this object;
        // the worker's shared_ptr keeps Impl alive until this function returns.
        if (worker.joinable() && worker.get_id() == std::this_thread::get_id()) {
            worker.detach();
            worker_id = {};
        }
        condition.notify_all();
    }

    void
    CreateSnapshot() {
        std::lock_guard<std::mutex> lock(mutex);
        if (!config.enabled) {
            return;
        }
        StoredSnapshot stored;
        auto& snapshot = stored.snapshot;
        snapshot.end_time = NowMillis();
        snapshot.timestamp = last_snapshot_end == 0 || last_snapshot_end > snapshot.end_time
                                 ? snapshot.end_time - config.heartbeat_interval_ms
                                 : last_snapshot_end;
        last_snapshot_end = snapshot.end_time;
        for (auto& entry : collectors) {
            if (entry.second.global.requests == 0) {
                continue;
            }
            TelemetryOperationMetrics operation;
            operation.operation = entry.first;
            std::vector<double> quantile_samples;
            operation.global = entry.second.global.Snapshot(&quantile_samples);
            stored.global_samples.emplace(entry.first, std::move(quantile_samples));
            for (const auto& collection : entry.second.collections) {
                if (all_collections_enabled || enabled_collections.count(collection.first) > 0) {
                    operation.collection_metrics.emplace(collection.first, collection.second.Snapshot());
                }
            }
            snapshot.metrics.emplace_back(std::move(operation));
            entry.second = OperationCollector{};
        }
        // Heartbeats report exactly the collector interval that just ended.
        // Keep this separate from retained history so skipping an empty history
        // entry cannot make the next heartbeat resend an older non-empty sample.
        latest_snapshot = snapshot;
        const auto history_start = snapshot.end_time - kMaxHistoryRangeMs;
        while (!snapshots.empty() && snapshots.front().snapshot.end_time < history_start) {
            snapshots.pop_front();
        }
        if (snapshot.metrics.empty()) {
            return;
        }
        snapshots.push_back(std::move(stored));
        while (snapshots.size() > kSnapshotLimit) {
            snapshots.pop_front();
        }
    }

    void
    SendHeartbeat() {
        proto::milvus::ClientHeartbeatRequest request;
        std::shared_ptr<proto::milvus::ClientTelemetryService::Stub> heartbeat_stub;
        uint64_t heartbeat_generation = 0;
        size_t reply_count = 0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stub == nullptr) {
                return;
            }
            auto* info = request.mutable_client_info();
            info->set_sdk_type("CPP");
            info->set_sdk_version(sdk_version);
            info->set_local_time(LocalTimeString());
            info->set_user(username);
            info->set_host(HostName());
            (*info->mutable_reserved())["client_id"] = client_id;
            (*info->mutable_reserved())["client_id_stable"] = stable_client_id ? "true" : "false";
            if (!database.empty()) {
                (*info->mutable_reserved())["db_name"] = database;
            }
            request.set_report_timestamp(NowMillis());
            // Do not resend the final enabled snapshot after collection is disabled. Replies,
            // config hash, cursor and incoming commands remain active as the control plane.
            if (config.enabled) {
                for (const auto& operation : latest_snapshot.metrics) {
                    auto* output = request.add_metrics();
                    output->set_operation(operation.operation);
                    *output->mutable_global() = ToProtoMetric(operation.global);
                    for (const auto& collection : operation.collection_metrics) {
                        if (all_collections_enabled || enabled_collections.count(collection.first) > 0) {
                            (*output->mutable_collection_metrics())[collection.first] =
                                ToProtoMetric(collection.second);
                        }
                    }
                }
            }
            for (const auto& reply : pending_replies) {
                auto* output = request.add_command_replies();
                output->set_command_id(reply.command_id);
                output->set_success(reply.success);
                output->set_error_message(reply.error_message);
                output->set_payload(reply.payload);
            }
            reply_count = pending_replies.size();
            request.set_config_hash(config_hash);
            request.set_last_command_timestamp(last_command_timestamp);
            heartbeat_stub = stub;
            heartbeat_generation = channel_generation;
        }

        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
        proto::milvus::ClientHeartbeatResponse response;
        auto grpc_status = heartbeat_stub->ClientHeartbeat(&context, request, &response);
        if (!grpc_status.ok()) {
            std::lock_guard<std::mutex> lock(mutex);
            if (heartbeat_generation != channel_generation) {
                return;
            }
            last_heartbeat_error = grpc_status.error_message();
            if (grpc_status.error_code() == grpc::StatusCode::UNIMPLEMENTED) {
                ++unsupported_streak;
            }
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (heartbeat_generation != channel_generation) {
                return;
            }
            // Reaching any server implementation proves the RPC exists, even when the
            // response carries a business error.
            unsupported_streak = 0;
            if (response.status().code() != 0 || response.status().error_code() != proto::common::ErrorCode::Success) {
                last_heartbeat_error = response.status().reason();
                return;
            }
        }
        std::vector<TelemetryCommand> commands;
        commands.reserve(response.commands_size());
        for (const auto& command : response.commands()) {
            commands.push_back({command.command_id(), command.command_type(), command.payload(), command.create_time(),
                                command.persistent(), command.target_scope()});
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (heartbeat_generation != channel_generation) {
                return;
            }
            pending_replies.erase(pending_replies.begin(),
                                  pending_replies.begin() + std::min(reply_count, pending_replies.size()));
            last_heartbeat_error.clear();
        }
        ProcessCommands(commands, heartbeat_generation);
    }

    TelemetryCommandReply
    HandleCommand(const TelemetryCommand& command) {
        CommandHandler handler;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto iterator = handlers.find(command.command_type);
            if (iterator == handlers.end()) {
                return FailedReply(command.command_id, "unknown command type: " + command.command_type);
            }
            handler = iterator->second;
        }
        try {
            return handler(command);
        } catch (const std::exception& exception) {
            return FailedReply(command.command_id, exception.what());
        } catch (...) {
            // Command handlers are extensible application code. Telemetry is best-effort and
            // must never terminate the process when a handler throws a non-standard exception.
            return FailedReply(command.command_id, "command handler threw a non-standard exception");
        }
    }

    void
    ProcessCommands(const std::vector<TelemetryCommand>& commands, uint64_t expected_generation = 0) {
        std::lock_guard<std::recursive_mutex> command_lock(command_mutex);
        int64_t previous_timestamp;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (expected_generation != 0 && expected_generation != channel_generation) {
                return;
            }
            previous_timestamp = last_command_timestamp;
        }
        int64_t max_timestamp = previous_timestamp;
        bool has_persistent = false;
        for (const auto& command : commands) {
            max_timestamp = std::max(max_timestamp, command.create_time);
            has_persistent = has_persistent || command.persistent;
            bool skip = false;
            {
                std::lock_guard<std::mutex> lock(mutex);
                skip = command.create_time < previous_timestamp || executed_commands.count(command.command_id) > 0;
                if (skip) {
                    pending_replies.push_back(SuccessReply(command.command_id));
                }
            }
            if (skip) {
                continue;
            }
            auto reply = HandleCommand(command);
            std::lock_guard<std::mutex> lock(mutex);
            executed_commands[command.command_id] = command.create_time;
            pending_replies.push_back(std::move(reply));
        }
        std::lock_guard<std::mutex> lock(mutex);
        for (auto iterator = executed_commands.begin(); iterator != executed_commands.end();) {
            if (iterator->second < max_timestamp) {
                iterator = executed_commands.erase(iterator);
            } else {
                ++iterator;
            }
        }
        if (has_persistent) {
            config_hash = ClientTelemetryManager::CalculateConfigHash(commands);
        }
        last_command_timestamp = std::max(last_command_timestamp, max_timestamp);
    }

    void
    RegisterDefaultHandlers() {
        handlers["push_config"] = [this](const TelemetryCommand& command) {
            auto payload = command.payload.empty() ? nlohmann::json::object() : nlohmann::json::parse(command.payload);
            if (!payload.is_object()) {
                throw std::invalid_argument("push_config payload must be a JSON object");
            }

            std::vector<std::string> applied;
            std::vector<std::string> ignored;
            bool enabled = false;
            int64_t interval = 0;
            double sampling_rate = 0;
            if (payload.count("enabled")) {
                if (!payload["enabled"].is_boolean()) {
                    throw std::invalid_argument("enabled must be a boolean");
                }
                enabled = payload["enabled"].get<bool>();
                applied.emplace_back("enabled");
            }
            if (payload.count("heartbeat_interval_ms")) {
                if (!payload["heartbeat_interval_ms"].is_number_integer() &&
                    !payload["heartbeat_interval_ms"].is_number_unsigned()) {
                    throw std::invalid_argument("heartbeat_interval_ms must be an integer");
                }
                interval = payload["heartbeat_interval_ms"].get<int64_t>();
                if (interval <= 0) {
                    throw std::invalid_argument("heartbeat_interval_ms must be positive");
                }
                applied.emplace_back("heartbeat_interval_ms");
            }
            if (payload.count("sampling_rate")) {
                if (!payload["sampling_rate"].is_number()) {
                    throw std::invalid_argument("sampling_rate must be a number");
                }
                sampling_rate = payload["sampling_rate"].get<double>();
                if (!std::isfinite(sampling_rate)) {
                    throw std::invalid_argument("sampling_rate must be finite");
                }
                sampling_rate = std::max(0.0, std::min(1.0, sampling_rate));
                applied.emplace_back("sampling_rate");
            }
            if (payload.count("ttl_seconds")) {
                if (!payload["ttl_seconds"].is_number_integer() && !payload["ttl_seconds"].is_number_unsigned()) {
                    throw std::invalid_argument("ttl_seconds must be an integer");
                }
                (void)payload["ttl_seconds"].get<int64_t>();
            }
            for (auto iterator = payload.begin(); iterator != payload.end(); ++iterator) {
                if (iterator.key() != "enabled" && iterator.key() != "heartbeat_interval_ms" &&
                    iterator.key() != "sampling_rate") {
                    ignored.push_back(iterator.key());
                }
            }
            std::sort(ignored.begin(), ignored.end());
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (payload.count("enabled")) {
                    config.enabled = enabled;
                }
                if (payload.count("heartbeat_interval_ms")) {
                    config.heartbeat_interval_ms = static_cast<uint64_t>(interval);
                }
                if (payload.count("sampling_rate")) {
                    config.sampling_rate = sampling_rate;
                }
                condition.notify_all();
            }
            return SuccessReply(command.command_id, nlohmann::json{{"applied", applied}, {"ignored", ignored}}.dump());
        };
        handlers["collection_metrics"] = [this](const TelemetryCommand& command) {
            std::lock_guard<std::mutex> lock(mutex);
            if (command.payload.empty()) {
                std::vector<std::string> names(enabled_collections.begin(), enabled_collections.end());
                std::sort(names.begin(), names.end());
                nlohmann::json result = {{"enabled_collections", names},
                                         {"all_collections_enabled", all_collections_enabled}};
                return SuccessReply(command.command_id, result.dump());
            }
            auto payload = nlohmann::json::parse(command.payload);
            if (!payload.is_object()) {
                throw std::invalid_argument("collection_metrics payload must be a JSON object");
            }
            if (payload.count("enabled") && !payload["enabled"].is_boolean()) {
                throw std::invalid_argument("enabled must be a boolean");
            }
            if (payload.count("collections") && !payload["collections"].is_array()) {
                throw std::invalid_argument("collections must be an array");
            }
            if (payload.count("metrics_types") && !payload["metrics_types"].is_array()) {
                throw std::invalid_argument("metrics_types must be an array");
            }
            bool enabled = payload.value("enabled", false);
            auto collections = payload.value("collections", std::vector<std::string>{});
            if (payload.count("metrics_types")) {
                (void)payload["metrics_types"].get<std::vector<std::string>>();
            }
            bool wildcard = std::find(collections.begin(), collections.end(), "*") != collections.end();
            if (enabled) {
                if (collections.empty()) {
                    throw std::invalid_argument("collections list cannot be empty when enabled=true");
                }
                if (wildcard) {
                    all_collections_enabled = true;
                } else {
                    enabled_collections.insert(collections.begin(), collections.end());
                }
            } else if (wildcard || collections.empty()) {
                all_collections_enabled = false;
                enabled_collections.clear();
            } else {
                for (const auto& collection : collections) {
                    enabled_collections.erase(collection);
                }
            }
            return SuccessReply(command.command_id);
        };
        handlers["show_errors"] = [this](const TelemetryCommand& command) {
            auto payload = command.payload.empty() ? nlohmann::json::object() : nlohmann::json::parse(command.payload);
            if (!payload.is_object()) {
                throw std::invalid_argument("show_errors payload must be a JSON object");
            }
            if (payload.count("max_count") && !payload["max_count"].is_number_integer() &&
                !payload["max_count"].is_number_unsigned()) {
                throw std::invalid_argument("max_count must be an integer");
            }
            auto requested = payload.value("max_count", int64_t{100});
            auto max_count = static_cast<size_t>(requested <= 0 ? 100 : requested);
            std::vector<TelemetryError> values;
            {
                std::lock_guard<std::mutex> lock(mutex);
                for (auto iterator = errors.rbegin(); iterator != errors.rend() && values.size() < max_count;
                     ++iterator) {
                    values.push_back(*iterator);
                }
            }
            if (values.empty()) {
                return SuccessReply(command.command_id);
            }
            nlohmann::json result = nlohmann::json::array();
            for (const auto& error : values) {
                nlohmann::json detail = {
                    {"timestamp", error.timestamp}, {"operation", error.operation}, {"error_msg", error.error_message}};
                if (!error.collection.empty()) {
                    detail["collection"] = error.collection;
                }
                if (!error.request_id.empty()) {
                    detail["request_id"] = error.request_id;
                }
                result.push_back(std::move(detail));
            }
            while (result.dump().size() > kMaxReplyBytes && result.size() > 1) {
                result.erase(result.begin() + result.size() / 2, result.end());
            }
            auto encoded = result.dump();
            while (encoded.size() > kMaxReplyBytes && result.size() == 1 &&
                   result.at(0).value("error_msg", std::string{}).size() > 1) {
                auto message = result.at(0).at("error_msg").get<std::string>();
                result.at(0)["error_msg"] =
                    message.substr(0, std::max<size_t>(1, message.size() / 2)) + "...(truncated)";
                encoded = result.dump();
            }
            if (encoded.size() > kMaxReplyBytes) {
                throw std::invalid_argument("show_errors response exceeds the 1MB payload limit");
            }
            return SuccessReply(command.command_id, encoded);
        };
        handlers["get_config"] = [this](const TelemetryCommand& command) {
            std::lock_guard<std::mutex> lock(mutex);
            std::vector<std::string> collections(enabled_collections.begin(), enabled_collections.end());
            std::sort(collections.begin(), collections.end());
            nlohmann::json user_config = {
                {"address", uri},
                {"username", username},
                {"db_name", database},
                {"telemetry_enabled", config.enabled},
                {"telemetry_heartbeat_interval_ms", config.heartbeat_interval_ms},
                {"telemetry_sampling_rate", config.sampling_rate},
                {"enabled_collections", all_collections_enabled ? std::vector<std::string>{"*"} : collections},
                {"all_collections_enabled", all_collections_enabled}};
            return SuccessReply(command.command_id, nlohmann::json{{"user_config", user_config}}.dump());
        };
        handlers["show_latency_history"] = [this](const TelemetryCommand& command) {
            if (command.payload.empty()) {
                throw std::invalid_argument("payload is required with start_time and end_time");
            }
            auto payload = nlohmann::json::parse(command.payload);
            if (!payload.is_object()) {
                throw std::invalid_argument("show_latency_history payload must be a JSON object");
            }
            if (payload.count("detail") && !payload["detail"].is_boolean()) {
                throw std::invalid_argument("detail must be a boolean");
            }
            int64_t start = ParseRfc3339Millis(payload.at("start_time").get<std::string>());
            int64_t end = ParseRfc3339Millis(payload.at("end_time").get<std::string>());
            if (end < start) {
                throw std::invalid_argument("end_time must be after start_time");
            }
            if (end - start > 60 * 60 * 1000) {
                throw std::invalid_argument("time range cannot exceed 1 hour");
            }
            std::vector<StoredSnapshot> selected;
            {
                std::lock_guard<std::mutex> lock(mutex);
                selected.reserve(snapshots.size());
                for (const auto& stored : snapshots) {
                    const auto& snapshot = stored.snapshot;
                    if (snapshot.end_time >= start && snapshot.timestamp <= end) {
                        selected.push_back(stored);
                    }
                }
            }
            nlohmann::json response;
            if (payload.value("detail", false)) {
                response["snapshots"] = nlohmann::json::array();
                for (const auto& stored : selected) {
                    const auto& snapshot = stored.snapshot;
                    nlohmann::json metrics = nlohmann::json::object();
                    for (const auto& operation : snapshot.metrics) {
                        metrics[operation.operation] = MetricJson(operation.global);
                    }
                    response["snapshots"].push_back(
                        {{"timestamp", snapshot.timestamp}, {"end_time", snapshot.end_time}, {"metrics", metrics}});
                }
                response["total_snapshots"] = selected.size();
            } else {
                struct Total {
                    int64_t requests{0};
                    int64_t successes{0};
                    int64_t errors{0};
                    double average{0};
                    double maximum{0};
                    struct WeightedSamples {
                        const std::vector<double>* values;
                        double weight;
                    };
                    std::vector<WeightedSamples> latency_windows;
                };
                std::map<std::string, Total> totals;
                for (const auto& stored : selected) {
                    const auto& snapshot = stored.snapshot;
                    for (const auto& operation : snapshot.metrics) {
                        auto& total = totals[operation.operation];
                        total.requests += operation.global.request_count;
                        total.successes += operation.global.success_count;
                        total.errors += operation.global.error_count;
                        total.average += operation.global.avg_latency_ms * operation.global.request_count;
                        total.maximum = std::max(total.maximum, operation.global.max_latency_ms);
                        const auto samples = stored.global_samples.find(operation.operation);
                        if (samples != stored.global_samples.end() && !samples->second.empty()) {
                            const auto weight =
                                static_cast<double>(operation.global.request_count) / samples->second.size();
                            total.latency_windows.push_back({&samples->second, weight});
                        }
                    }
                }
                nlohmann::json metrics = nlohmann::json::object();
                for (const auto& entry : totals) {
                    struct Cursor {
                        double latency;
                        size_t window;
                        size_t sample;
                    };
                    const auto later = [](const Cursor& left, const Cursor& right) {
                        return left.latency > right.latency;
                    };
                    std::priority_queue<Cursor, std::vector<Cursor>, decltype(later)> samples(later);
                    for (size_t window = 0; window < entry.second.latency_windows.size(); ++window) {
                        const auto* values = entry.second.latency_windows[window].values;
                        if (values != nullptr && !values->empty()) {
                            samples.push({values->front(), window, 0});
                        }
                    }
                    double p99 = 0;
                    if (!samples.empty()) {
                        const auto target = static_cast<double>(entry.second.requests) * 0.99;
                        double cumulative = 0;
                        while (!samples.empty()) {
                            const auto sample = samples.top();
                            samples.pop();
                            const auto& window = entry.second.latency_windows[sample.window];
                            p99 = sample.latency;
                            cumulative += window.weight;
                            if (cumulative > target) {
                                break;
                            }
                            const auto next = sample.sample + 1;
                            if (next < window.values->size()) {
                                samples.push({(*window.values)[next], sample.window, next});
                            }
                        }
                    }
                    metrics[entry.first] = {
                        {"request_count", entry.second.requests},
                        {"success_count", entry.second.successes},
                        {"error_count", entry.second.errors},
                        {"avg_latency_ms",
                         entry.second.requests == 0 ? 0 : entry.second.average / entry.second.requests},
                        {"p99_latency_ms", p99},
                        {"max_latency_ms", entry.second.maximum}};
                }
                response = {{"aggregated", {{"start_time", start}, {"end_time", end}, {"metrics", metrics}}},
                            {"snapshot_count", selected.size()}};
            }
            auto encoded = response.dump();
            if (encoded.size() > kMaxReplyBytes) {
                throw std::invalid_argument("response too large, try a smaller time range");
            }
            return SuccessReply(command.command_id, encoded);
        };
    }

    mutable std::mutex mutex;
    std::condition_variable condition;
    TelemetryConfig config;
    const TelemetryConfig connection_config;
    const bool stable_client_id;
    const std::string client_id;
    std::shared_ptr<proto::milvus::ClientTelemetryService::Stub> stub;
    uint64_t channel_generation{0};
    std::string username;
    std::string database;
    std::string uri;
    std::string sdk_version;
    std::string connection_scope;
    std::unordered_map<std::string, OperationCollector> collectors;
    std::deque<TelemetryError> errors;
    TelemetrySnapshot latest_snapshot;
    std::deque<StoredSnapshot> snapshots;
    std::vector<TelemetryCommandReply> pending_replies;
    std::unordered_map<std::string, int64_t> executed_commands;
    std::unordered_map<std::string, CommandHandler> handlers;
    std::unordered_set<std::string> enabled_collections;
    bool all_collections_enabled{false};
    bool ready{false};
    bool stopped{true};
    bool control_plane_activated{false};
    bool worker_running{false};
    bool join_in_progress{false};
    bool external_stop_requested{false};
    int unsupported_streak{0};
    // Carries the fractional sampling rate between operations, in kSamplingScale units:
    // each operation adds the rate and the one that pushes it past a whole unit is the one
    // sampled.
    uint64_t sampling_accum{0};
    int64_t last_command_timestamp{0};
    int64_t last_snapshot_end{0};
    std::string config_hash;
    std::string last_heartbeat_error;
    std::thread worker;
    std::thread::id worker_id;
    std::recursive_mutex command_mutex;
};

ClientTelemetryManager::ClientTelemetryManager(const TelemetryConfig& config, const std::string& runtime_client_id)
    : impl_(std::make_shared<Impl>(config, runtime_client_id)) {
}

ClientTelemetryManager::~ClientTelemetryManager() {
    // The worker also owns Impl while it is running. Always request shutdown
    // before releasing the manager's reference so destruction from inside a
    // command handler cannot free state that HeartbeatLoop is still using.
    impl_->Stop();
}

void
ClientTelemetryManager::AttachChannel(const std::shared_ptr<grpc::Channel>& channel, const std::string& username,
                                      const std::string& database, const std::string& uri,
                                      const std::string& sdk_version, const std::string& connection_scope) {
    impl_->AttachChannel(channel, username, database, uri, sdk_version, connection_scope);
}

void
ClientTelemetryManager::Start() {
    impl_->Start();
}

void
ClientTelemetryManager::Stop() {
    impl_->Stop();
}

bool
ClientTelemetryManager::IsReady() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->ready;
}

bool
ClientTelemetryManager::isWorkerThread() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->worker_running && impl_->worker_id == std::this_thread::get_id();
}

bool
ClientTelemetryManager::IsSupported() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->unsupported_streak == 0;
}

const std::string&
ClientTelemetryManager::ClientId() const {
    return impl_->client_id;
}

std::string
ClientTelemetryManager::ConfigHash() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->config_hash;
}

int64_t
ClientTelemetryManager::LastCommandTimestamp() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->last_command_timestamp;
}

TelemetryConfig
ClientTelemetryManager::Config() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->config;
}

bool
ClientTelemetryManager::MatchesConnection(const TelemetryConfig& config, const std::string& connection_scope) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->connection_scope == connection_scope &&
           SameTelemetryConfig(impl_->connection_config, NormalizedTelemetryConfig(config));
}

std::string
ClientTelemetryManager::LastHeartbeatError() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->last_heartbeat_error;
}

void
ClientTelemetryManager::RegisterCommandHandler(const std::string& command_type, CommandHandler handler) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->handlers[command_type] = std::move(handler);
}

void
ClientTelemetryManager::RecordOperation(const std::string& operation, const google::protobuf::Message& request,
                                        std::chrono::steady_clock::time_point started, bool success,
                                        const std::string& error_message, const std::string& request_id) {
    RecordOperation(operation, operation == "RunAnalyzer" ? std::string{} : CollectionName(request), started, success,
                    error_message, request_id);
}

void
ClientTelemetryManager::RecordOperation(const std::string& operation, const std::string& collection,
                                        std::chrono::steady_clock::time_point started, bool success,
                                        const std::string& error_message, const std::string& request_id) {
    static const std::unordered_set<std::string> operations = {"Insert",       "Delete", "Upsert",     "Search",
                                                               "HybridSearch", "Query",  "RunAnalyzer"};
    if (operations.count(operation) == 0) {
        return;
    }
    auto latency_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - started).count();
    auto latency = static_cast<double>(latency_us) / 1000.0;
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->config.enabled) {
        return;
    }
    auto rate = impl_->config.sampling_rate;
    bool sampled = rate >= 1.0;
    if (rate > 0.0 && rate < 1.0) {
        // Sample on the operation that carries the accumulator across a whole unit, so the
        // sampled operations are spread evenly: at 0.25 that is every fourth one. The
        // ratio has to hold over any stretch of operations, not only over a long one --
        // metrics are reported per heartbeat window, and a window is tens or hundreds of
        // operations, so sampling a contiguous run would make each window either complete
        // or empty. A rate too small to represent still samples rarely rather than never.
        auto step = static_cast<uint64_t>(rate * kSamplingScale);
        if (step == 0) {
            step = 1;
        }
        auto before = impl_->sampling_accum;
        impl_->sampling_accum = before + step;
        sampled = impl_->sampling_accum / kSamplingScale != before / kSamplingScale;
    }
    if (!sampled) {
        return;
    }
    bool collection_enabled = impl_->all_collections_enabled || impl_->enabled_collections.count(collection) > 0;
    auto& collector = impl_->collectors[operation];
    collector.global.Record(latency, success);
    if (!collection.empty() && collection_enabled) {
        collector.collections[collection].Record(latency, success);
    }
    if (!success) {
        impl_->errors.push_back({NowMillis(), operation, error_message, collection,
                                 ClientRequestContext::IsValid(request_id) ? request_id : std::string{}});
        while (impl_->errors.size() > impl_->config.error_max_count) {
            impl_->errors.pop_front();
        }
    }
}

std::vector<TelemetryError>
ClientTelemetryManager::RecentErrors(size_t max_count) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::vector<TelemetryError> result;
    for (auto iterator = impl_->errors.rbegin(); iterator != impl_->errors.rend() && result.size() < max_count;
         ++iterator) {
        result.push_back(*iterator);
    }
    return result;
}

std::vector<TelemetrySnapshot>
ClientTelemetryManager::MetricsSnapshots() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::vector<TelemetrySnapshot> result;
    result.reserve(impl_->snapshots.size());
    for (const auto& stored : impl_->snapshots) {
        result.push_back(stored.snapshot);
    }
    return result;
}

std::vector<TelemetryCommandReply>
ClientTelemetryManager::PendingCommandReplies() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->pending_replies;
}

void
ClientTelemetryManager::ProcessCommands(const std::vector<TelemetryCommand>& commands) {
    impl_->ProcessCommands(commands);
}

std::string
ClientTelemetryManager::CalculateConfigHash(const std::vector<TelemetryCommand>& commands) {
    std::vector<TelemetryCommand> persistent;
    for (const auto& command : commands) {
        if (command.persistent) {
            persistent.push_back(command);
        }
    }
    if (persistent.empty()) {
        return "";
    }
    std::sort(persistent.begin(), persistent.end(), [](const TelemetryCommand& left, const TelemetryCommand& right) {
        return left.command_id < right.command_id;
    });
    Sha256 hash;
    for (const auto& command : persistent) {
        hash.Update(command.command_id);
        hash.Update(command.command_type);
        hash.Update(command.payload);
    }
    return hash.Finish().substr(0, 16);
}

}  // namespace milvus
