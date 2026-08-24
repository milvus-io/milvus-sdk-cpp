// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "cpp-httplib/httplib.h"
#include "milvus.grpc.pb.h"
#include "milvus.pb.h"
#include "milvus/ClientTelemetry.h"
#include "milvus/thirdparty/nlohmann/json.hpp"
#include "milvus/types/ConnectParam.h"
#include "types/GlobalCluster.h"
#include "utils/ConnectionHandler.h"
#include "utils/GlobalClusterUtils.h"
#include "utils/TopologyRefresher.h"

namespace {

std::string
TopologyBody(int64_t version, const std::string& endpoint = "a:19530") {
    return R"({"code":0,"data":{"version":)" + std::to_string(version) +
           R"(,"clusters":[{"clusterId":"a","endpoint":")" + endpoint + R"(","capability":3}]}})";
}

class GlobalMilvusService final : public milvus::proto::milvus::MilvusService::Service {
 public:
    grpc::Status
    Connect(grpc::ServerContext*, const milvus::proto::milvus::ConnectRequest*,
            milvus::proto::milvus::ConnectResponse*) override {
        ++connects;
        return grpc::Status::OK;
    }

    std::atomic<int> connects{0};
};

std::unique_ptr<grpc::Server>
StartGlobalMilvusServer(GlobalMilvusService& service, int& port) {
    grpc::ServerBuilder builder;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    return builder.BuildAndStart();
}

class ScopedTopologyServerThread {
 public:
    explicit ScopedTopologyServerThread(httplib::Server& server)
        : server_(server), thread_([&server]() { server.listen_after_bind(); }) {
    }

    ~ScopedTopologyServerThread() {
        server_.stop();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    ScopedTopologyServerThread(const ScopedTopologyServerThread&) = delete;
    ScopedTopologyServerThread&
    operator=(const ScopedTopologyServerThread&) = delete;

 private:
    httplib::Server& server_;
    std::thread thread_;
};

}  // namespace

TEST(GlobalClusterUtilsTest, IsGlobalEndpoint) {
    EXPECT_TRUE(milvus::GlobalClusterUtils::IsGlobalEndpoint("https://xxx.global-cluster.yyy.com:443"));
    EXPECT_TRUE(milvus::GlobalClusterUtils::IsGlobalEndpoint("http://GLOBAL-CLUSTER.local"));
    EXPECT_FALSE(milvus::GlobalClusterUtils::IsGlobalEndpoint(""));
    EXPECT_FALSE(milvus::GlobalClusterUtils::IsGlobalEndpoint("http://localhost:19530"));
}

TEST(GlobalClusterUtilsTest, BuildTopologyUrl) {
    EXPECT_EQ(milvus::GlobalClusterUtils::BuildTopologyUrl("https://xxx.global-cluster.yyy.com:443"),
              "https://xxx.global-cluster.yyy.com:443/global-cluster/topology");
    EXPECT_EQ(milvus::GlobalClusterUtils::BuildTopologyUrl("xxx.global-cluster.yyy.com"),
              "https://xxx.global-cluster.yyy.com/global-cluster/topology");
    // an explicit http scheme is preserved (matching pymilvus)
    EXPECT_EQ(milvus::GlobalClusterUtils::BuildTopologyUrl("http://host.global-cluster.com/"),
              "http://host.global-cluster.com/global-cluster/topology");
}

TEST(GlobalClusterUtilsTest, BuildPrimaryUri) {
    // https global endpoint -> https primary
    EXPECT_EQ(milvus::GlobalClusterUtils::BuildPrimaryUri("https://x.global-cluster.com:443", false, "p:19530"),
              "https://p:19530");
    // http global endpoint, no tls -> http primary
    EXPECT_EQ(milvus::GlobalClusterUtils::BuildPrimaryUri("http://x.global-cluster.com:19530", false, "p:19530"),
              "http://p:19530");
    // http global endpoint but TLS enabled -> https primary
    EXPECT_EQ(milvus::GlobalClusterUtils::BuildPrimaryUri("http://x.global-cluster.com:19530", true, "p:19530"),
              "https://p:19530");
    // scheme-less global endpoint defaults to https, matching BuildTopologyUrl()
    EXPECT_EQ(milvus::GlobalClusterUtils::BuildPrimaryUri("x.global-cluster.com", false, "p:19530"), "https://p:19530");
    // explicit scheme in the primary endpoint is used as-is
    EXPECT_EQ(
        milvus::GlobalClusterUtils::BuildPrimaryUri("http://x.global-cluster.com:19530", false, "https://p:19530"),
        "https://p:19530");
}

TEST(GlobalClusterUtilsTest, ParseTopologyResponse) {
    const std::string body = R"({
      "code": 0,
      "data": {
        "version": 3,
        "clusters": [
          {"clusterId": "c1", "endpoint": "host1:19530", "capability": 1},
          {"clusterId": "c2", "endpoint": "host2:19530", "capability": 3}
        ]
      }
    })";
    milvus::GlobalTopology topology;
    auto status = milvus::GlobalClusterUtils::ParseTopologyResponse(body, topology);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    EXPECT_EQ(topology.Version(), 3);
    ASSERT_EQ(topology.Clusters().size(), 2u);
    EXPECT_EQ(topology.Clusters()[0].ClusterId(), "c1");
    EXPECT_EQ(topology.Clusters()[0].Endpoint(), "host1:19530");
    EXPECT_EQ(topology.Clusters()[0].Capability(), 1);
    EXPECT_FALSE(topology.Clusters()[0].IsPrimary());
    EXPECT_TRUE(topology.Clusters()[1].IsPrimary());
    ASSERT_NE(topology.Primary(), nullptr);
    EXPECT_EQ(topology.Primary()->Endpoint(), "host2:19530");
}

TEST(GlobalClusterUtilsTest, ParseTopologyResponseStringVersion) {
    // the server may return data.version as a numeric string (pymilvus uses "version": "1")
    const std::string body =
        R"({"code":0,"data":{"version":"1","clusters":[{"clusterId":"a","endpoint":"a:19530","capability":3}]}})";
    milvus::GlobalTopology topology;
    auto status = milvus::GlobalClusterUtils::ParseTopologyResponse(body, topology);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    EXPECT_EQ(topology.Version(), 1);
}

TEST(GlobalClusterUtilsTest, ParseTopologyResponseErrors) {
    milvus::GlobalTopology topology;
    auto status = milvus::GlobalClusterUtils::ParseTopologyResponse("not json", topology);
    EXPECT_FALSE(status.IsOk());

    status = milvus::GlobalClusterUtils::ParseTopologyResponse(R"({"code": 1, "message": "boom"})", topology);
    EXPECT_EQ(status.Code(), milvus::StatusCode::SERVER_FAILED);

    status = milvus::GlobalClusterUtils::ParseTopologyResponse(R"({"code": 0})", topology);
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);

    status = milvus::GlobalClusterUtils::ParseTopologyResponse(R"({"code":0,"data":{"version":1,"clusters":[{}]}})",
                                                               topology);
    EXPECT_EQ(status.Code(), milvus::StatusCode::INVALID_ARGUMENT);
}

TEST(GlobalClusterUtilsTest, FetchTopology) {
    httplib::Server server;
    const std::string body = R"({"code":0,"data":{"version":7,"clusters":[
      {"clusterId":"a","endpoint":"a:19530","capability":3}]}})";
    server.Get("/global-cluster/topology",
               [body](const httplib::Request&, httplib::Response& res) { res.set_content(body, "application/json"); });
    auto port = server.bind_to_any_port("127.0.0.1");
    ASSERT_TRUE(server.is_valid());
    std::thread srv([&server]() { server.listen_after_bind(); });
    server.wait_until_ready();

    milvus::GlobalTopology topology;
    auto status =
        milvus::GlobalClusterUtils::FetchTopology("http://127.0.0.1:" + std::to_string(port), "tok", topology);
    EXPECT_TRUE(status.IsOk()) << status.Message();
    if (status.IsOk()) {
        EXPECT_EQ(topology.Version(), 7);
        ASSERT_EQ(topology.Clusters().size(), 1u);
        EXPECT_EQ(topology.Clusters()[0].Endpoint(), "a:19530");
    }

    server.stop();
    srv.join();
}

TEST(GlobalClusterUtilsTest, FetchTopologyRetriesOnMalformedBody) {
    httplib::Server server;
    std::atomic<int> requests{0};
    const std::string bad_body = R"({"code":0,"data":{"version":7,"clusters":[)";
    const std::string good_body = TopologyBody(7);
    server.Get("/global-cluster/topology", [&](const httplib::Request&, httplib::Response& res) {
        // first response is truncated/malformed, subsequent responses are valid
        res.set_content(requests.fetch_add(1) == 0 ? bad_body : good_body, "application/json");
    });
    auto port = server.bind_to_any_port("127.0.0.1");
    ASSERT_TRUE(server.is_valid());
    std::thread srv([&server]() { server.listen_after_bind(); });
    server.wait_until_ready();

    milvus::GlobalTopology topology;
    auto status =
        milvus::GlobalClusterUtils::FetchTopology("http://127.0.0.1:" + std::to_string(port), "tok", topology);
    EXPECT_TRUE(status.IsOk()) << status.Message();
    if (status.IsOk()) {
        EXPECT_EQ(topology.Version(), 7);
    }
    EXPECT_EQ(requests.load(), 2);

    server.stop();
    srv.join();
}

TEST(GlobalClusterUtilsTest, FetchTopologyRetriesOnNon200) {
    httplib::Server server;
    std::atomic<int> requests{0};
    const std::string good_body = TopologyBody(7);
    server.Get("/global-cluster/topology", [&](const httplib::Request&, httplib::Response& res) {
        if (requests.fetch_add(1) == 0) {
            res.status = 503;
            res.set_content("temporarily unavailable", "text/plain");
        } else {
            res.set_content(good_body, "application/json");
        }
    });
    auto port = server.bind_to_any_port("127.0.0.1");
    ASSERT_TRUE(server.is_valid());
    std::thread srv([&server]() { server.listen_after_bind(); });
    server.wait_until_ready();

    milvus::GlobalTopology topology;
    auto status =
        milvus::GlobalClusterUtils::FetchTopology("http://127.0.0.1:" + std::to_string(port), "tok", topology);
    EXPECT_TRUE(status.IsOk()) << status.Message();
    if (status.IsOk()) {
        EXPECT_EQ(topology.Version(), 7);
    }
    EXPECT_EQ(requests.load(), 2);

    server.stop();
    srv.join();
}

TEST(GlobalClusterUtilsTest, FetchTopologyAbortsOnShouldStop) {
    httplib::Server server;
    std::atomic<int> requests{0};
    const std::string good_body = TopologyBody(7);
    server.Get("/global-cluster/topology", [&](const httplib::Request&, httplib::Response& res) {
        requests.fetch_add(1);
        res.set_content(good_body, "application/json");
    });
    auto port = server.bind_to_any_port("127.0.0.1");
    ASSERT_TRUE(server.is_valid());
    std::thread srv([&server]() { server.listen_after_bind(); });
    server.wait_until_ready();

    milvus::GlobalTopology topology;
    // should_stop returns true immediately: the fetch must abort before issuing any request
    auto status = milvus::GlobalClusterUtils::FetchTopology("http://127.0.0.1:" + std::to_string(port), "tok", topology,
                                                            []() { return true; });
    EXPECT_FALSE(status.IsOk());
    EXPECT_EQ(requests.load(), 0);

    server.stop();
    srv.join();
}

TEST(TopologyRefresherTest, CallbackOnVersionChange) {
    httplib::Server server;
    std::atomic<int64_t> served_version{1};
    server.Get("/global-cluster/topology", [&served_version](const httplib::Request&, httplib::Response& res) {
        res.set_content(TopologyBody(served_version.load()), "application/json");
    });
    auto port = server.bind_to_any_port("127.0.0.1");
    ASSERT_TRUE(server.is_valid());
    std::thread srv([&server]() { server.listen_after_bind(); });
    server.wait_until_ready();

    std::atomic<int> calls{0};
    int64_t last_version = 0;
    milvus::TopologyRefresher refresher("http://127.0.0.1:" + std::to_string(port), "tok", 1,
                                        std::chrono::milliseconds(50),
                                        [&](const milvus::GlobalTopology& t, const std::function<bool()>&) {
                                            last_version = t.Version();
                                            calls.fetch_add(1);
                                            return true;
                                        });
    refresher.Start();

    // version 1 == initial version, no callback expected yet
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    EXPECT_EQ(calls.load(), 0);

    // bump the version, the next refresh should invoke the callback
    served_version.store(2);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    refresher.Stop();

    EXPECT_GT(calls.load(), 0);
    EXPECT_EQ(last_version, 2);

    server.stop();
    srv.join();
}

TEST(GlobalClusterTelemetryTest, FailoverAndUseDatabasePreserveLogicalTelemetryState) {
    GlobalMilvusService first_service;
    GlobalMilvusService second_service;
    int first_port = 0;
    int second_port = 0;
    auto first_server = StartGlobalMilvusServer(first_service, first_port);
    auto second_server = StartGlobalMilvusServer(second_service, second_port);
    ASSERT_NE(first_server, nullptr);
    ASSERT_NE(second_server, nullptr);

    httplib::Server topology_server;
    std::atomic<int64_t> served_version{1};
    std::atomic<int> served_primary_port{first_port};
    // Put the global-cluster marker in the path rather than a localhost subdomain. Windows does
    // not consistently resolve arbitrary *.localhost names, while a numeric loopback address is
    // portable across all CI runners.
    topology_server.Get(
        "/global-cluster-test/global-cluster/topology", [&](const httplib::Request&, httplib::Response& response) {
            response.set_content(
                TopologyBody(served_version.load(), "127.0.0.1:" + std::to_string(served_primary_port.load())),
                "application/json");
        });
    const auto topology_port = topology_server.bind_to_any_port("127.0.0.1");
    ASSERT_TRUE(topology_server.is_valid());
    ScopedTopologyServerThread topology_thread(topology_server);
    topology_server.wait_until_ready();

    const auto logical_endpoint = "http://127.0.0.1:" + std::to_string(topology_port) + "/global-cluster-test";
    milvus::ConnectParam connect_param(logical_endpoint);
    connect_param.SetDbName("primary_db");
    milvus::TelemetryConfig telemetry_config;
    telemetry_config.enabled = false;
    connect_param.SetTelemetryConfig(telemetry_config);

    milvus::ConnectionHandler handler;
    ASSERT_TRUE(handler.Connect(connect_param).IsOk());
    auto manager = handler.GetTelemetry();
    ASSERT_NE(manager, nullptr);
    const auto client_id = manager->ClientId();
    manager->ProcessCommands({
        {"config", "push_config", R"({"sampling_rate":0.25})", 1, true, ""},
        {"collections", "collection_metrics", R"({"enabled":true,"collections":["before_failover"]})", 2, false, ""},
    });
    const auto config_hash = manager->ConfigHash();
    const auto reply_count = manager->PendingCommandReplies().size();
    ASSERT_EQ(reply_count, 2U);

    served_primary_port = second_port;
    served_version = 2;
    handler.TriggerGlobalRefresh();
    const auto second_uri = "http://127.0.0.1:" + std::to_string(second_port);
    for (int retry = 0; retry < 3000; ++retry) {
        auto connection = handler.GetConnection();
        if (connection != nullptr && connection->GetConnectParam().Uri() == second_uri) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_NE(handler.GetConnection(), nullptr);
    EXPECT_EQ(handler.GetConnection()->GetConnectParam().Uri(), second_uri);
    EXPECT_EQ(handler.CurrentEndpoint(), logical_endpoint);
    EXPECT_EQ(handler.GetTelemetry(), manager);
    EXPECT_EQ(manager->ClientId(), client_id);
    EXPECT_EQ(manager->ConfigHash(), config_hash);
    EXPECT_EQ(manager->LastCommandTimestamp(), 2);
    EXPECT_DOUBLE_EQ(manager->Config().sampling_rate, 0.25);
    EXPECT_EQ(manager->PendingCommandReplies().size(), reply_count);

    ASSERT_TRUE(handler.UseDatabase("secondary_db").IsOk());
    ASSERT_EQ(handler.GetTelemetry(), manager);
    manager->ProcessCommands({{"config-after-use-db", "get_config", "", 3, false, ""}});
    const auto replies = manager->PendingCommandReplies();
    ASSERT_FALSE(replies.empty());
    ASSERT_TRUE(replies.back().success) << replies.back().error_message;
    const auto user_config = nlohmann::json::parse(replies.back().payload).at("user_config");
    EXPECT_EQ(user_config.at("address"), logical_endpoint);
    EXPECT_EQ(user_config.at("db_name"), "secondary_db");
    EXPECT_EQ(manager->ClientId(), client_id);
    EXPECT_EQ(manager->ConfigHash(), config_hash);

    EXPECT_TRUE(handler.Disconnect().IsOk());
    first_server->Shutdown();
    second_server->Shutdown();
}
