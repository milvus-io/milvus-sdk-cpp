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

#include <grpcpp/client_context.h>
#include <grpcpp/channel.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>
#include <milvus/thirdparty/nlohmann/json.hpp>

#include "common.pb.h"
#include "milvus.grpc.pb.h"
#include "milvus.pb.h"

namespace milvus {
namespace {

constexpr size_t kSampleBufferSize = 1000;
constexpr size_t kSnapshotLimit = 120;
constexpr uint64_t kSamplingDenominator = 10000;
constexpr size_t kMaxReplyBytes = 1024 * 1024;
constexpr uint64_t kMaxUnsupportedBackoffMs = 30 * 60 * 1000;

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
    stream << std::hex << std::setfill('0') << std::setw(8) << static_cast<uint32_t>(high >> 32) << "-"
           << std::setw(4) << static_cast<uint16_t>(high >> 16) << "-" << std::setw(4)
           << static_cast<uint16_t>(high) << "-" << std::setw(4) << static_cast<uint16_t>(low >> 48) << "-"
           << std::setw(12) << (low & 0x0000FFFFFFFFFFFFULL);
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
    const char* value = std::getenv("HOSTNAME");
    return value == nullptr ? "Unknown" : value;
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
    Sha256()
        : state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19} {
    }

    void
    Update(const std::string& value) {
        Update(reinterpret_cast<const uint8_t*>(value.data()), value.size());
    }

    std::string
    Finish() {
        uint64_t bit_length = total_size_ * 8;
        buffer_[buffer_size_++] = 0x80;
        if (buffer_size_ > 56) {
            while (buffer_size_ < 64) {
                buffer_[buffer_size_++] = 0;
            }
            Transform(buffer_.data());
            buffer_size_ = 0;
        }
        while (buffer_size_ < 56) {
            buffer_[buffer_size_++] = 0;
        }
        for (int index = 7; index >= 0; --index) {
            buffer_[buffer_size_++] = static_cast<uint8_t>(bit_length >> (index * 8));
        }
        Transform(buffer_.data());

        std::ostringstream stream;
        stream << std::hex << std::setfill('0');
        for (auto value : state_) {
            stream << std::setw(8) << value;
        }
        return stream.str();
    }

 private:
    void
    Update(const uint8_t* data, size_t size) {
        total_size_ += size;
        for (size_t index = 0; index < size; ++index) {
            buffer_[buffer_size_++] = data[index];
            if (buffer_size_ == 64) {
                Transform(buffer_.data());
                buffer_size_ = 0;
            }
        }
    }

    void
    Transform(const uint8_t* block) {
        static const uint32_t constants[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2};
        uint32_t schedule[64];
        for (size_t index = 0; index < 16; ++index) {
            schedule[index] = (static_cast<uint32_t>(block[index * 4]) << 24) |
                              (static_cast<uint32_t>(block[index * 4 + 1]) << 16) |
                              (static_cast<uint32_t>(block[index * 4 + 2]) << 8) |
                              static_cast<uint32_t>(block[index * 4 + 3]);
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
Timegm(std::tm* value) {
#ifdef _WIN32
    return static_cast<int64_t>(_mkgmtime(value));
#else
    return static_cast<int64_t>(timegm(value));
#endif
}

int64_t
ParseRfc3339Millis(const std::string& value) {
    if (value.size() < 19) {
        throw std::invalid_argument("invalid RFC3339 timestamp");
    }
    std::tm time{};
    std::istringstream stream(value.substr(0, 19));
    stream >> std::get_time(&time, "%Y-%m-%dT%H:%M:%S");
    if (stream.fail()) {
        throw std::invalid_argument("invalid RFC3339 timestamp");
    }
    size_t position = 19;
    int64_t milliseconds = 0;
    if (position < value.size() && value[position] == '.') {
        size_t end = position + 1;
        while (end < value.size() && std::isdigit(static_cast<unsigned char>(value[end]))) {
            ++end;
        }
        auto fraction = value.substr(position + 1, end - position - 1);
        while (fraction.size() < 3) {
            fraction.push_back('0');
        }
        milliseconds = std::stoll(fraction.substr(0, 3));
        position = end;
    }
    int offset_seconds = 0;
    if (position < value.size() && value[position] != 'Z') {
        if (position + 5 >= value.size() || (value[position] != '+' && value[position] != '-')) {
            throw std::invalid_argument("invalid RFC3339 timezone");
        }
        int sign = value[position] == '+' ? 1 : -1;
        int hours = std::stoi(value.substr(position + 1, 2));
        int minutes = std::stoi(value.substr(position + 4, 2));
        offset_seconds = sign * (hours * 3600 + minutes * 60);
    }
    return (Timegm(&time) - offset_seconds) * 1000 + milliseconds;
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
    Snapshot() const {
        std::vector<double> sorted(samples.begin(), samples.end());
        std::sort(sorted.begin(), sorted.end());
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
    return {{"request_count", metric.request_count},
            {"success_count", metric.success_count},
            {"error_count", metric.error_count},
            {"avg_latency_ms", metric.avg_latency_ms},
            {"p99_latency_ms", metric.p99_latency_ms},
            {"max_latency_ms", metric.max_latency_ms}};
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

class ClientTelemetryManager::Impl {
 public:
    Impl(const TelemetryConfig& value, const std::string& runtime_client_id)
        : config(value),
          stable_client_id(!value.client_id.empty()),
          client_id(stable_client_id ? value.client_id
                                     : (runtime_client_id.empty() ? RandomUuid() : runtime_client_id)) {
        if (config.heartbeat_interval_ms == 0) {
            config.heartbeat_interval_ms = 30000;
        }
        config.sampling_rate = std::max(0.0, std::min(1.0, config.sampling_rate));
        if (config.error_max_count == 0) {
            config.error_max_count = 100;
        }
        RegisterDefaultHandlers();
    }

    ~Impl() {
        Stop();
    }

    void
    AttachChannel(const std::shared_ptr<grpc::Channel>& channel, const std::string& user, const std::string& db,
                  const std::string& endpoint, const std::string& version) {
        std::lock_guard<std::mutex> lock(mutex);
        stub = proto::milvus::ClientTelemetryService::NewStub(channel);
        username = user;
        database = db;
        uri = endpoint;
        sdk_version = version;
    }

    void
    Start() {
        std::lock_guard<std::mutex> lock(mutex);
        if (ready) {
            return;
        }
        ready = true;
        stopped = false;
        if (!config.enabled) {
            return;
        }
        worker = std::thread([this]() { HeartbeatLoop(); });
    }

    void
    Stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!ready && stopped) {
                return;
            }
            stopped = true;
            condition.notify_all();
        }
        if (worker.joinable() && worker.get_id() != std::this_thread::get_id()) {
            worker.join();
        }
        std::lock_guard<std::mutex> lock(mutex);
        ready = false;
    }

    void
    HeartbeatLoop() {
        while (true) {
            CreateSnapshot();
            SendHeartbeat();
            std::unique_lock<std::mutex> lock(mutex);
            if (stopped) {
                return;
            }
            uint64_t delay = config.heartbeat_interval_ms;
            if (unsupported_streak > 0) {
                auto exponent = std::min(unsupported_streak, 20);
                delay = std::min(kMaxUnsupportedBackoffMs, delay * (uint64_t{1} << exponent));
            }
            condition.wait_for(lock, std::chrono::milliseconds(delay), [this]() { return stopped; });
            if (stopped) {
                return;
            }
        }
    }

    void
    CreateSnapshot() {
        std::lock_guard<std::mutex> lock(mutex);
        if (!config.enabled) {
            return;
        }
        TelemetrySnapshot snapshot;
        snapshot.end_time = NowMillis();
        snapshot.timestamp = last_snapshot_end == 0 ? snapshot.end_time - config.heartbeat_interval_ms : last_snapshot_end;
        last_snapshot_end = snapshot.end_time;
        for (auto& entry : collectors) {
            if (entry.second.global.requests == 0) {
                continue;
            }
            TelemetryOperationMetrics operation;
            operation.operation = entry.first;
            operation.global = entry.second.global.Snapshot();
            for (const auto& collection : entry.second.collections) {
                operation.collection_metrics.emplace(collection.first, collection.second.Snapshot());
            }
            snapshot.metrics.emplace_back(std::move(operation));
            entry.second = OperationCollector{};
        }
        snapshots.push_back(std::move(snapshot));
        while (snapshots.size() > kSnapshotLimit) {
            snapshots.pop_front();
        }
    }

    void
    SendHeartbeat() {
        proto::milvus::ClientHeartbeatRequest request;
        std::unique_ptr<proto::milvus::ClientTelemetryService::Stub>* stub_pointer = nullptr;
        size_t reply_count = 0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (!config.enabled || stub == nullptr) {
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
            if (!snapshots.empty()) {
                for (const auto& operation : snapshots.back().metrics) {
                    auto* output = request.add_metrics();
                    output->set_operation(operation.operation);
                    *output->mutable_global() = ToProtoMetric(operation.global);
                    for (const auto& collection : operation.collection_metrics) {
                        (*output->mutable_collection_metrics())[collection.first] = ToProtoMetric(collection.second);
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
            stub_pointer = &stub;
        }

        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
        proto::milvus::ClientHeartbeatResponse response;
        auto grpc_status = (*stub_pointer)->ClientHeartbeat(&context, request, &response);
        if (!grpc_status.ok()) {
            std::lock_guard<std::mutex> lock(mutex);
            last_heartbeat_error = grpc_status.error_message();
            if (grpc_status.error_code() == grpc::StatusCode::UNIMPLEMENTED) {
                ++unsupported_streak;
            }
            return;
        }
        if (response.status().code() != 0 || response.status().error_code() != proto::common::ErrorCode::Success) {
            std::lock_guard<std::mutex> lock(mutex);
            last_heartbeat_error = response.status().reason();
            return;
        }
        std::vector<TelemetryCommand> commands;
        commands.reserve(response.commands_size());
        for (const auto& command : response.commands()) {
            commands.push_back({command.command_id(), command.command_type(), command.payload(), command.create_time(),
                                command.persistent(), command.target_scope()});
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            pending_replies.erase(pending_replies.begin(),
                                  pending_replies.begin() + std::min(reply_count, pending_replies.size()));
            unsupported_streak = 0;
            last_heartbeat_error.clear();
        }
        ProcessCommands(commands);
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
        }
    }

    void
    ProcessCommands(const std::vector<TelemetryCommand>& commands) {
        int64_t previous_timestamp;
        {
            std::lock_guard<std::mutex> lock(mutex);
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
            if (iterator->second <= previous_timestamp) {
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
            std::lock_guard<std::mutex> lock(mutex);
            if (payload.count("enabled")) {
                config.enabled = payload["enabled"].get<bool>();
            }
            if (payload.count("heartbeat_interval_ms")) {
                auto interval = payload["heartbeat_interval_ms"].get<int64_t>();
                if (interval <= 0) {
                    throw std::invalid_argument("heartbeat_interval_ms must be positive");
                }
                config.heartbeat_interval_ms = static_cast<uint64_t>(interval);
            }
            if (payload.count("sampling_rate")) {
                config.sampling_rate = std::max(0.0, std::min(1.0, payload["sampling_rate"].get<double>()));
            }
            condition.notify_all();
            return SuccessReply(command.command_id);
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
            bool enabled = payload.value("enabled", false);
            auto collections = payload.value("collections", std::vector<std::string>{});
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
            auto max_count = payload.value("max_count", static_cast<size_t>(100));
            std::vector<TelemetryError> values;
            {
                std::lock_guard<std::mutex> lock(mutex);
                for (auto iterator = errors.rbegin(); iterator != errors.rend() && values.size() < max_count;
                     ++iterator) {
                    values.push_back(*iterator);
                }
            }
            nlohmann::json result = nlohmann::json::array();
            for (const auto& error : values) {
                result.push_back({{"timestamp", error.timestamp},
                                  {"operation", error.operation},
                                  {"error_msg", error.error_message},
                                  {"collection", error.collection},
                                  {"request_id", error.request_id}});
            }
            while (result.dump().size() > kMaxReplyBytes && result.size() > 1) {
                result.erase(result.begin() + result.size() / 2, result.end());
            }
            auto encoded = result.dump();
            while (encoded.size() > kMaxReplyBytes && result.size() == 1 &&
                   result.at(0).value("error_msg", std::string{}).size() > 1) {
                auto message = result.at(0).at("error_msg").get<std::string>();
                result.at(0)["error_msg"] = message.substr(0, std::max<size_t>(1, message.size() / 2)) +
                                               "...(truncated)";
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
            nlohmann::json user_config = {{"address", uri},
                                          {"username", username},
                                          {"db_name", database},
                                          {"telemetry_enabled", config.enabled},
                                          {"telemetry_heartbeat_interval_ms", config.heartbeat_interval_ms},
                                          {"telemetry_sampling_rate", config.sampling_rate},
                                          {"enabled_collections",
                                           all_collections_enabled ? std::vector<std::string>{"*"} : collections},
                                          {"all_collections_enabled", all_collections_enabled}};
            return SuccessReply(command.command_id, nlohmann::json{{"user_config", user_config}}.dump());
        };
        handlers["show_latency_history"] = [this](const TelemetryCommand& command) {
            if (command.payload.empty()) {
                throw std::invalid_argument("payload is required with start_time and end_time");
            }
            auto payload = nlohmann::json::parse(command.payload);
            int64_t start = ParseRfc3339Millis(payload.at("start_time").get<std::string>());
            int64_t end = ParseRfc3339Millis(payload.at("end_time").get<std::string>());
            if (end < start) {
                throw std::invalid_argument("end_time must be after start_time");
            }
            if (end - start > 60 * 60 * 1000) {
                throw std::invalid_argument("time range cannot exceed 1 hour");
            }
            std::vector<TelemetrySnapshot> all;
            {
                std::lock_guard<std::mutex> lock(mutex);
                all.assign(snapshots.begin(), snapshots.end());
            }
            std::vector<TelemetrySnapshot> selected;
            for (const auto& snapshot : all) {
                if (snapshot.end_time >= start && snapshot.timestamp <= end) {
                    selected.push_back(snapshot);
                }
            }
            nlohmann::json response;
            if (payload.value("detail", false)) {
                response["snapshots"] = nlohmann::json::array();
                for (const auto& snapshot : selected) {
                    nlohmann::json metrics;
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
                    double p99{0};
                    double maximum{0};
                };
                std::map<std::string, Total> totals;
                for (const auto& snapshot : selected) {
                    for (const auto& operation : snapshot.metrics) {
                        auto& total = totals[operation.operation];
                        total.requests += operation.global.request_count;
                        total.successes += operation.global.success_count;
                        total.errors += operation.global.error_count;
                        total.average += operation.global.avg_latency_ms * operation.global.request_count;
                        total.p99 += operation.global.p99_latency_ms * operation.global.request_count;
                        total.maximum = std::max(total.maximum, operation.global.max_latency_ms);
                    }
                }
                nlohmann::json metrics;
                for (const auto& entry : totals) {
                    metrics[entry.first] = {{"request_count", entry.second.requests},
                                            {"success_count", entry.second.successes},
                                            {"error_count", entry.second.errors},
                                            {"avg_latency_ms", entry.second.requests == 0
                                                                   ? 0
                                                                   : entry.second.average / entry.second.requests},
                                            {"p99_latency_ms", entry.second.requests == 0
                                                                   ? 0
                                                                   : entry.second.p99 / entry.second.requests},
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
    const bool stable_client_id;
    const std::string client_id;
    std::unique_ptr<proto::milvus::ClientTelemetryService::Stub> stub;
    std::string username;
    std::string database;
    std::string uri;
    std::string sdk_version;
    std::unordered_map<std::string, OperationCollector> collectors;
    std::deque<TelemetryError> errors;
    std::deque<TelemetrySnapshot> snapshots;
    std::vector<TelemetryCommandReply> pending_replies;
    std::unordered_map<std::string, int64_t> executed_commands;
    std::unordered_map<std::string, CommandHandler> handlers;
    std::unordered_set<std::string> enabled_collections;
    bool all_collections_enabled{false};
    bool ready{false};
    bool stopped{true};
    int unsupported_streak{0};
    uint64_t sampling_counter{0};
    int64_t last_command_timestamp{0};
    int64_t last_snapshot_end{0};
    std::string config_hash;
    std::string last_heartbeat_error;
    std::thread worker;
};

ClientTelemetryManager::ClientTelemetryManager(const TelemetryConfig& config, const std::string& runtime_client_id)
    : impl_(new Impl(config, runtime_client_id)) {
}

ClientTelemetryManager::~ClientTelemetryManager() = default;

void
ClientTelemetryManager::AttachChannel(const std::shared_ptr<grpc::Channel>& channel, const std::string& username,
                                      const std::string& database, const std::string& uri,
                                      const std::string& sdk_version) {
    impl_->AttachChannel(channel, username, database, uri, sdk_version);
}

void
ClientTelemetryManager::UpdateDatabase(const std::string& database) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->database = database;
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
    static const std::unordered_set<std::string> operations = {"Insert", "Delete", "Upsert", "Search",
                                                                "HybridSearch", "Query", "RunAnalyzer"};
    if (operations.count(operation) == 0) {
        return;
    }
    auto latency = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    auto collection = CollectionName(request);
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->config.enabled) {
        return;
    }
    auto rate = impl_->config.sampling_rate;
    bool sampled = rate >= 1.0;
    if (rate > 0.0 && rate < 1.0) {
        auto threshold = static_cast<uint64_t>(rate * kSamplingDenominator);
        sampled = threshold > 0 && ++impl_->sampling_counter % kSamplingDenominator < threshold;
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
        impl_->errors.push_back({NowMillis(), operation, error_message, collection, request_id});
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
    return {impl_->snapshots.begin(), impl_->snapshots.end()};
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
    std::sort(persistent.begin(), persistent.end(),
              [](const TelemetryCommand& left, const TelemetryCommand& right) {
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
