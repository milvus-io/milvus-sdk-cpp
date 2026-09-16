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

#include <gtest/gtest.h>

#include <memory>

#include "../mocks/MilvusMockedTest.h"
#include "milvus/MilvusClientV2.h"

using ::milvus::proto::milvus::ConnectRequest;
using ::milvus::proto::milvus::ConnectResponse;
using ::milvus::proto::milvus::CreateResourceGroupRequest;
using ::milvus::proto::milvus::DescribeResourceGroupRequest;
using ::milvus::proto::milvus::DescribeResourceGroupResponse;
using ::milvus::proto::milvus::DropResourceGroupRequest;
using ::milvus::proto::milvus::ListResourceGroupsRequest;
using ::milvus::proto::milvus::ListResourceGroupsResponse;
using ::milvus::proto::milvus::TransferNodeRequest;
using ::milvus::proto::milvus::TransferReplicaRequest;
using ::milvus::proto::milvus::UpdateResourceGroupsRequest;
using ::testing::_;

namespace {

milvus::MilvusClientV2Ptr
CreateConnectedV2Client(testing::StrictMock<::milvus::MilvusMockedService>& service, uint16_t port) {
    EXPECT_CALL(service, Connect(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ConnectRequest*, ConnectResponse*) { return ::grpc::Status{}; });

    auto client = milvus::MilvusClientV2::Create();
    auto status = client->Connect(milvus::ConnectParam{"127.0.0.1", port});
    EXPECT_TRUE(status.IsOk());
    return client;
}

}  // namespace

TEST_F(UnconnectMilvusMockedTest, CreateResourceGroupSendsNameAndConfig) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, CreateResourceGroup(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const CreateResourceGroupRequest* request, ::milvus::proto::common::Status*) {
                EXPECT_EQ(request->resource_group(), "rg1");
                EXPECT_EQ(request->config().requests().node_num(), 2);
                EXPECT_EQ(request->config().limits().node_num(), 4);
                EXPECT_EQ(request->config().transfer_from_size(), 1);
                EXPECT_EQ(request->config().transfer_from(0).resource_group(), "rg2");
                EXPECT_EQ(request->config().transfer_to_size(), 1);
                EXPECT_EQ(request->config().transfer_to(0).resource_group(), "rg3");
                EXPECT_EQ(request->config().node_filter().node_labels_size(), 1);
                EXPECT_EQ(request->config().node_filter().node_labels(0).key(), "CPU");
                EXPECT_EQ(request->config().node_filter().node_labels(0).value(), "32");
                return ::grpc::Status{};
            });

    milvus::ResourceGroupConfig config;
    config.SetRequests(2);
    config.SetLimits(4);
    config.AddTrnasferFromGroup("rg2");
    config.AddTrnasferToGroup("rg3");
    config.AddNodeFilter("CPU", "32");
    milvus::CreateResourceGroupRequest request;
    request.WithName("rg1").WithConfig(std::move(config));

    auto status = client->CreateResourceGroup(request);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, DropResourceGroupSendsGroupName) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DropResourceGroup(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const DropResourceGroupRequest* request, ::milvus::proto::common::Status*) {
                EXPECT_EQ(request->resource_group(), "rg1");
                return ::grpc::Status{};
            });

    milvus::DropResourceGroupRequest request;
    request.WithGroupName("rg1");
    auto status = client->DropResourceGroup(request);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, UpdateResourceGroupsSendsConfigMap) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, UpdateResourceGroups(_, _, _))
        .WillOnce(
            [](::grpc::ServerContext*, const UpdateResourceGroupsRequest* request, ::milvus::proto::common::Status*) {
                EXPECT_EQ(request->resource_groups_size(), 1);
                EXPECT_EQ(request->resource_groups().at("rg1").requests().node_num(), 2);
                EXPECT_EQ(request->resource_groups().at("rg1").limits().node_num(), 4);
                return ::grpc::Status{};
            });

    milvus::ResourceGroupConfig config;
    config.SetRequests(2);
    config.SetLimits(4);
    std::unordered_map<std::string, milvus::ResourceGroupConfig> groups{{"rg1", std::move(config)}};
    milvus::UpdateResourceGroupsRequest request;
    request.WithGroups(std::move(groups));

    auto status = client->UpdateResourceGroups(request);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, TransferNodeSendsSourceTargetAndCount) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, TransferNode(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const TransferNodeRequest* request, ::milvus::proto::common::Status*) {
            EXPECT_EQ(request->source_resource_group(), "rg1");
            EXPECT_EQ(request->target_resource_group(), "rg2");
            EXPECT_EQ(request->num_node(), 5);
            return ::grpc::Status{};
        });

    milvus::TransferNodeRequest request;
    request.WithSourceGroup("rg1").WithTargetGroup("rg2").WithNumNodes(5);
    auto status = client->TransferNode(request);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, TransferReplicaSendsCollectionAndCount) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, TransferReplica(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const TransferReplicaRequest* request, ::milvus::proto::common::Status*) {
            EXPECT_EQ(request->source_resource_group(), "rg1");
            EXPECT_EQ(request->target_resource_group(), "rg2");
            EXPECT_EQ(request->collection_name(), "coll");
            EXPECT_EQ(request->num_replica(), 3);
            return ::grpc::Status{};
        });

    milvus::TransferReplicaRequest request;
    request.WithSourceGroup("rg1").WithTargetGroup("rg2").WithCollectionName("coll").WithNumReplicas(3);
    auto status = client->TransferReplica(request);
    EXPECT_TRUE(status.IsOk());
}

TEST_F(UnconnectMilvusMockedTest, ListResourceGroupsParsesGroupNames) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, ListResourceGroups(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const ListResourceGroupsRequest*, ListResourceGroupsResponse* response) {
            response->add_resource_groups("rg1");
            response->add_resource_groups("rg2");
            return ::grpc::Status{};
        });

    milvus::ListResourceGroupsRequest request;
    milvus::ListResourceGroupsResponse response;
    auto status = client->ListResourceGroups(request, response);

    EXPECT_TRUE(status.IsOk());
    EXPECT_EQ(response.GroupNames(), (std::vector<std::string>{"rg1", "rg2"}));
}

TEST_F(UnconnectMilvusMockedTest, DescribeResourceGroupParsesResourceGroupDesc) {
    auto client = CreateConnectedV2Client(service_, server_.ListenPort());

    EXPECT_CALL(service_, DescribeResourceGroup(_, _, _))
        .WillOnce([](::grpc::ServerContext*, const DescribeResourceGroupRequest* request,
                     DescribeResourceGroupResponse* response) {
            EXPECT_EQ(request->resource_group(), "rg1");

            auto* group = response->mutable_resource_group();
            group->set_name("rg1");
            group->set_capacity(4);
            group->set_num_available_node(2);
            (*group->mutable_num_loaded_replica())["coll1"] = 1;
            (*group->mutable_num_outgoing_node())["rg2"] = 1;
            (*group->mutable_num_incoming_node())["rg3"] = 1;
            group->mutable_config()->mutable_requests()->set_node_num(2);
            group->mutable_config()->mutable_limits()->set_node_num(4);
            auto* node = group->add_nodes();
            node->set_node_id(100);
            node->set_address("addr");
            node->set_hostname("host");
            return ::grpc::Status{};
        });

    milvus::DescribeResourceGroupRequest request;
    request.WithGroupName("rg1");
    milvus::DescribeResourceGroupResponse response;
    auto status = client->DescribeResourceGroup(request, response);

    EXPECT_TRUE(status.IsOk());
    const auto& desc = response.Desc();
    EXPECT_EQ(desc.Name(), "rg1");
    EXPECT_EQ(desc.Capacity(), 4);
    EXPECT_EQ(desc.AvailableNodesNum(), 2);
    ASSERT_EQ(desc.LoadedReplicasNum().size(), 1);
    EXPECT_EQ(desc.LoadedReplicasNum().at("coll1"), 1);
    ASSERT_EQ(desc.OutgoingNodesNum().size(), 1);
    EXPECT_EQ(desc.OutgoingNodesNum().at("rg2"), 1);
    ASSERT_EQ(desc.IncomingNodesNum().size(), 1);
    EXPECT_EQ(desc.IncomingNodesNum().at("rg3"), 1);
    EXPECT_EQ(desc.Config().Requests(), 2);
    EXPECT_EQ(desc.Config().Limits(), 4);
    ASSERT_EQ(desc.Nodes().size(), 1);
    EXPECT_EQ(desc.Nodes()[0].id_, 100);
    EXPECT_EQ(desc.Nodes()[0].address_, "addr");
    EXPECT_EQ(desc.Nodes()[0].hostname_, "host");
}
