// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "milvus/ClientRequestContext.h"
#include "milvus/ClientTelemetry.h"
#include "milvus/MilvusClient.h"
#include "milvus/MilvusClientV2.h"
#include "milvus/thirdparty/nlohmann/json.hpp"

namespace {

using Json = nlohmann::json;

const std::string kTelemetryHost = std::getenv("MILVUS_TELEMETRY_HOST") == nullptr
                                       ? "127.0.0.1"
                                       : std::getenv("MILVUS_TELEMETRY_HOST");
const uint16_t kTelemetryPort = static_cast<uint16_t>(
    std::getenv("MILVUS_TELEMETRY_PORT") == nullptr ? 9091 : std::stoi(std::getenv("MILVUS_TELEMETRY_PORT")));
const std::string kTelemetryBase = "/api/v1/_telemetry";

void
Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string
ToLower(std::string value) {
    for (auto& character : value) {
        if (character >= 'A' && character <= 'Z') {
            character = static_cast<char>(character - 'A' + 'a');
        }
    }
    return value;
}

std::string
DecodeChunked(const std::string& body) {
    std::string decoded;
    size_t offset = 0;
    while (offset < body.size()) {
        auto line_end = body.find("\r\n", offset);
        Require(line_end != std::string::npos, "Malformed chunked HTTP response");
        auto size_text = body.substr(offset, line_end - offset);
        auto extension = size_text.find(';');
        if (extension != std::string::npos) {
            size_text.resize(extension);
        }
        size_t chunk_size = std::stoul(size_text, nullptr, 16);
        offset = line_end + 2;
        if (chunk_size == 0) {
            break;
        }
        Require(offset + chunk_size <= body.size(), "Truncated chunked HTTP response");
        decoded.append(body, offset, chunk_size);
        offset += chunk_size + 2;
    }
    return decoded;
}

std::string
HttpRequest(const std::string& method, const std::string& path, const std::string& body = "") {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* addresses = nullptr;
    auto port = std::to_string(kTelemetryPort);
    int lookup = getaddrinfo(kTelemetryHost.c_str(), port.c_str(), &hints, &addresses);
    Require(lookup == 0, std::string("getaddrinfo failed: ") + gai_strerror(lookup));

    int socket_fd = -1;
    for (auto* address = addresses; address != nullptr; address = address->ai_next) {
        socket_fd = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (socket_fd >= 0 && connect(socket_fd, address->ai_addr, address->ai_addrlen) == 0) {
            break;
        }
        if (socket_fd >= 0) {
            close(socket_fd);
        }
        socket_fd = -1;
    }
    freeaddrinfo(addresses);
    Require(socket_fd >= 0, "Unable to connect to Milvus telemetry HTTP endpoint");

    std::ostringstream request;
    request << method << " " << path << " HTTP/1.1\r\n"
            << "Host: " << kTelemetryHost << ':' << kTelemetryPort << "\r\n"
            << "Accept: application/json\r\n"
            << "Connection: close\r\n";
    if (!body.empty()) {
        request << "Content-Type: application/json\r\n" << "Content-Length: " << body.size() << "\r\n";
    }
    request << "\r\n" << body;
    auto wire = request.str();
    size_t sent = 0;
    while (sent < wire.size()) {
        auto count = send(socket_fd, wire.data() + sent, wire.size() - sent, 0);
        if (count <= 0) {
            close(socket_fd);
            throw std::runtime_error("Failed to send telemetry HTTP request");
        }
        sent += static_cast<size_t>(count);
    }

    std::string response;
    char buffer[8192];
    while (true) {
        auto count = recv(socket_fd, buffer, sizeof(buffer), 0);
        if (count < 0) {
            close(socket_fd);
            throw std::runtime_error("Failed to read telemetry HTTP response");
        }
        if (count == 0) {
            break;
        }
        response.append(buffer, static_cast<size_t>(count));
    }
    close(socket_fd);

    auto headers_end = response.find("\r\n\r\n");
    Require(headers_end != std::string::npos, "Malformed telemetry HTTP response");
    auto headers = response.substr(0, headers_end);
    auto status_end = headers.find("\r\n");
    auto status_line = headers.substr(0, status_end);
    Require(status_line.find(" 200 ") != std::string::npos, "Telemetry HTTP request failed: " + status_line);
    auto response_body = response.substr(headers_end + 4);
    if (ToLower(headers).find("transfer-encoding: chunked") != std::string::npos) {
        response_body = DecodeChunked(response_body);
    }
    return response_body;
}

Json
ClientState(const std::string& client_id) {
    auto response = Json::parse(HttpRequest(
        "GET", kTelemetryBase + "/clients?client_id=" + client_id + "&include_metrics=true"));
    auto clients = response.value("clients", Json::array());
    return clients.empty() ? Json() : clients.at(0);
}

Json
WaitFor(const std::string& label, const std::string& client_id, const std::function<bool(const Json&)>& predicate) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
    Json last;
    while (std::chrono::steady_clock::now() < deadline) {
        last = ClientState(client_id);
        if (!last.is_null() && predicate(last)) {
            std::cout << "PASS " << label << std::endl;
            return last;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    throw std::runtime_error("Timed out waiting for " + label + "; last=" + last.dump());
}

std::string
PushCommand(const std::string& client_id, const std::string& command_type, const Json& payload,
            bool persistent = false) {
    Json request = {{"command_type", command_type},
                    {"target_client_id", client_id},
                    {"payload", payload},
                    {"ttl_seconds", 30},
                    {"persistent", persistent}};
    auto response = Json::parse(HttpRequest("POST", kTelemetryBase + "/commands", request.dump()));
    return response.at("command_id").get<std::string>();
}

Json
FindReply(const Json& state, const std::string& command_id) {
    auto replies = state.value("command_replies", Json::array());
    for (const auto& reply : replies) {
        if (reply.value("command_id", "") == command_id) {
            return reply;
        }
    }
    return Json();
}

Json
WaitForReply(const std::string& client_id, const std::string& command_id) {
    auto state = WaitFor("command reply " + command_id, client_id,
                         [&command_id](const Json& candidate) { return !FindReply(candidate, command_id).is_null(); });
    return FindReply(state, command_id);
}

bool
HasMetric(const Json& state, const std::string& operation, const std::string& counter, int64_t minimum,
          const std::string& collection = "") {
    auto metrics = state.value("metrics", Json::array());
    for (const auto& metric : metrics) {
        if (metric.value("operation", "") != operation) {
            continue;
        }
        auto global = metric.value("global", Json::object());
        if (global.value(counter, static_cast<int64_t>(0)) < minimum) {
            continue;
        }
        if (collection.empty()) {
            return true;
        }
        auto collections = metric.value("collection_metrics", Json::object());
        return collections.find(collection) != collections.end();
    }
    return false;
}

int
Run() {
    auto milvus_host = std::getenv("MILVUS_HOST") == nullptr ? "127.0.0.1" : std::getenv("MILVUS_HOST");
    auto milvus_port = static_cast<uint16_t>(
        std::getenv("MILVUS_PORT") == nullptr ? 19530 : std::stoi(std::getenv("MILVUS_PORT")));

    {
        auto legacy_client_id = "e2e-cpp-legacy-" + milvus::ClientRequestContext::NewRequestId();
        milvus::TelemetryConfig legacy_telemetry;
        legacy_telemetry.client_id = legacy_client_id;
        legacy_telemetry.heartbeat_interval_ms = 500;
        milvus::ConnectParam legacy_param{milvus_host, milvus_port};
        legacy_param.WithTelemetryConfig(legacy_telemetry);
        auto legacy_client = milvus::MilvusClient::Create();
        auto status = legacy_client->Connect(legacy_param);
        Require(status.IsOk(), "Legacy telemetry connect failed: " + status.Message());
        Require(legacy_client->GetTelemetry()->ClientId() == legacy_client_id,
                "Unexpected legacy telemetry client ID");
        WaitFor("legacy client registration", legacy_client_id,
                [](const Json& state) { return state.value("status", "") == "active"; });
        status = legacy_client->Disconnect();
        Require(status.IsOk(), "Legacy telemetry disconnect failed: " + status.Message());
    }

    {
        milvus::ConnectParam default_param{milvus_host, milvus_port};
        auto default_client = milvus::MilvusClientV2::Create();
        auto status = default_client->Connect(default_param);
        Require(status.IsOk(), "Default telemetry connect failed: " + status.Message());
        auto default_manager = default_client->GetTelemetry();
        Require(default_manager != nullptr, "Default telemetry manager is missing");
        Require(default_manager->Config().client_id.empty(), "Generated client ID was marked as stable");
        auto default_client_id = default_manager->ClientId();
        WaitFor("default client registration", default_client_id,
                [](const Json& state) { return state.value("status", "") == "active"; });

        status = default_client->UseDatabase("default");
        Require(status.IsOk(), "Default telemetry reconnect failed: " + status.Message());
        Require(default_client->GetTelemetry() == default_manager, "Reconnect replaced the telemetry manager");
        Require(default_manager->ClientId() == default_client_id, "Reconnect changed the runtime client ID");
        Require(default_manager->Config().client_id.empty(), "Reconnect marked the runtime client ID as stable");
        WaitFor("default client reconnect", default_client_id,
                [](const Json& state) { return state.value("status", "") == "active"; });
        status = default_client->Disconnect();
        Require(status.IsOk(), "Default telemetry disconnect failed: " + status.Message());
    }

    auto trace_suffix = milvus::ClientRequestContext::NewRequestId();
    auto client_id = "e2e-cpp-" + trace_suffix;
    milvus::TelemetryConfig telemetry_config;
    telemetry_config.client_id = client_id;
    telemetry_config.heartbeat_interval_ms = 500;
    telemetry_config.sampling_rate = 1.0;

    milvus::ConnectParam connect_param{milvus_host, milvus_port};
    connect_param.WithTelemetryConfig(telemetry_config);

    auto client = milvus::MilvusClientV2::Create();
    auto status = client->Connect(connect_param);
    Require(status.IsOk(), "Milvus connect failed: " + status.Message());
    try {
        auto manager = client->GetTelemetry();
        Require(manager != nullptr, "Telemetry manager is missing");
        Require(manager->ClientId() == client_id, "Unexpected telemetry client ID");
        WaitFor("client registration", client_id,
                [](const Json& state) { return state.value("status", "") == "active"; });

        milvus::RunAnalyzerRequest analyzer_request;
        analyzer_request.AddText("hello milvus telemetry")
            .WithAnalyzerParams({{"type", "standard"}})
            .WithDetail(true);
        milvus::RunAnalyzerResponse analyzer_response;
        status = client->RunAnalyzer(analyzer_request, analyzer_response);
        Require(status.IsOk(), "RunAnalyzer failed: " + status.Message());
        const auto& tokens = analyzer_response.Results().at(0).Tokens();
        Require(tokens.size() == 3 && tokens[0].token_ == "hello" && tokens[1].token_ == "milvus" &&
                    tokens[2].token_ == "telemetry",
                "Unexpected RunAnalyzer tokens");
        WaitFor("RunAnalyzer metric", client_id,
                [](const Json& state) { return HasMetric(state, "RunAnalyzer", "success_count", 1); });

        auto collection_command =
            PushCommand(client_id, "collection_metrics", {{"collections", {"*"}}, {"enabled", true}});
        Require(WaitForReply(client_id, collection_command).value("success", false),
                "collection_metrics command failed");

        auto request_id = milvus::ClientRequestContext::NewRequestId();
        {
            milvus::ScopedClientRequestId scoped(request_id);
            milvus::QueryRequest query_request;
            query_request.WithCollectionName("telemetry_e2e_missing").WithFilter("id > 0");
            milvus::QueryResponse query_response;
            status = client->Query(query_request, query_response);
        }
        Require(!status.IsOk(), "Query against missing collection unexpectedly succeeded");
        WaitFor("failed Query collection metric", client_id, [](const Json& state) {
            return HasMetric(state, "Query", "error_count", 1, "telemetry_e2e_missing");
        });

        auto errors_reply = WaitForReply(client_id, PushCommand(client_id, "show_errors", {{"max_count", 10}}));
        Require(errors_reply.value("success", false), "show_errors command failed");
        auto errors = Json::parse(errors_reply.at("payload").get<std::string>());
        bool trace_found = false;
        for (const auto& error : errors) {
            trace_found = trace_found || (error.value("operation", "") == "Query" &&
                                          error.value("request_id", "") == request_id);
        }
        Require(trace_found, "show_errors did not include the request ID");
        std::cout << "PASS request-id in show_errors" << std::endl;

        Json config_payload = {{"sampling_rate", 0.75}, {"heartbeat_interval_ms", 600}};
        auto config_command = PushCommand(client_id, "push_config", config_payload, true);
        Require(WaitForReply(client_id, config_command).value("success", false), "push_config command failed");
        milvus::TelemetryCommand expected_config{config_command, "push_config", config_payload.dump(), 0, true, ""};
        Require(manager->ConfigHash() == milvus::ClientTelemetryManager::CalculateConfigHash({expected_config}),
                "Persistent config hash mismatch");
        Require(manager->LastCommandTimestamp() > 0, "Command timestamp was not updated");

        auto get_config_reply = WaitForReply(client_id, PushCommand(client_id, "get_config", Json::object()));
        Require(get_config_reply.value("success", false), "get_config command failed");
        auto user_config = Json::parse(get_config_reply.at("payload").get<std::string>()).at("user_config");
        Require(user_config.value("telemetry_sampling_rate", 0.0) == 0.75, "Sampling rate was not applied");
        Require(user_config.value("telemetry_heartbeat_interval_ms", 0) == 600,
                "Heartbeat interval was not applied");
        Require(user_config.value("all_collections_enabled", false), "Collection metrics wildcard was not applied");
        std::cout << "CPP_E2E_OK " << client_id << std::endl;
    } catch (...) {
        client->Disconnect();
        throw;
    }
    status = client->Disconnect();
    Require(status.IsOk(), "Milvus disconnect failed: " + status.Message());
    return 0;
}

}  // namespace

int
main() {
    try {
        return Run();
    } catch (const std::exception& exception) {
        std::cerr << "CPP_E2E_FAILED: " << exception.what() << std::endl;
        return 1;
    }
}
