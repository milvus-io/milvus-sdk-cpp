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

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {

void
printRunningRequest(const milvus::RunningRequestInfo& info) {
    std::cout << "\tRequestId: " << info.RequestId() << std::endl;
    std::cout << "\tProxyId: " << info.ProxyId() << std::endl;
    std::cout << "\tType: " << info.Type() << std::endl;
    std::cout << "\tDatabaseName: " << info.DatabaseName() << std::endl;
    std::cout << "\tCollectionName: " << info.CollectionName() << std::endl;
    std::cout << "\tUser: " << info.User() << std::endl;
    std::cout << "\tClientAddr: " << info.ClientAddr() << std::endl;
    std::cout << "\tNq: " << info.Nq() << ", TopK: " << info.TopK() << std::endl;
    std::cout << "\tExpr: " << info.Expr() << std::endl;
    std::cout << "\tStartTimeMs: " << info.StartTimeMs() << std::endl;
    std::cout << "\tQueuedMs: " << info.QueuedMs() << ", ElapsedMs: " << info.ElapsedMs() << std::endl;
    std::cout << "\tState: " << info.State() << std::endl;
    std::cout << "\tTraceId: " << info.TraceId() << std::endl;
    std::cout << "\tCancellable: " << (info.Cancellable() ? "true" : "false") << std::endl;
    std::cout << "\tTaskIds:";
    for (int64_t task_id : info.TaskIds()) {
        std::cout << " " << task_id;
    }
    std::cout << std::endl;
}

void
printNodeResults(const milvus::RunningRequestNodeResults& node_results) {
    for (const auto& node_result : node_results) {
        std::cout << "\tNode " << node_result.NodeId() << ": error_code=" << node_result.ErrorCode() << ", reason=\""
                  << node_result.Reason() << "\", unimplemented=" << (node_result.Unimplemented() ? "true" : "false")
                  << std::endl;
    }
}

}  // namespace

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClientV2::Create();

    milvus::ConnectParam connect_param{"http://localhost:19530", "root:Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    // Optional filters, combined with AND; empty means no constraint. The user may list the
    // requests they issued themselves, and other users' requests with the ListRunningRequests
    // privilege.
    milvus::ListRunningRequestsRequest list_request;
    if (argc > 1 && argv[1][0] != '\0') {
        list_request.WithDatabaseName(argv[1]);
    }
    if (argc > 2 && argv[2][0] != '\0') {
        list_request.WithCollectionName(argv[2]);
    }

    milvus::ListRunningRequestsResponse list_response;
    status = client->ListRunningRequests(list_request, list_response);
    util::CheckStatus("list running requests", status);

    std::cout << "Running requests:" << std::endl;
    for (const auto& info : list_response.Requests()) {
        printRunningRequest(info);
        std::cout << "------------------------------" << std::endl;
    }
    std::cout << "Node results:" << std::endl;
    printNodeResults(list_response.NodeResults());

    // Cancel the requests reported above by passing their ids on the command line.
    if (argc > 3) {
        milvus::CancelRequestsRequest cancel_request;
        for (int i = 3; i < argc; ++i) {
            cancel_request.AddRequestId(argv[i]);
        }
        cancel_request.WithReason("canceled by running_requests example");

        milvus::CancelRequestsResponse cancel_response;
        status = client->CancelRequests(cancel_request, cancel_response);
        util::CheckStatus("cancel requests", status);

        std::cout << "Canceled requests:" << std::endl;
        for (const auto& info : cancel_response.Canceled()) {
            printRunningRequest(info);
            std::cout << "------------------------------" << std::endl;
        }
        std::cout << "Not found:";
        for (const auto& id : cancel_response.NotFound()) {
            std::cout << " " << id;
        }
        std::cout << std::endl;
        std::cout << "Undetermined:";
        for (const auto& id : cancel_response.Undetermined()) {
            std::cout << " " << id;
        }
        std::cout << std::endl;
        std::cout << "Node results:" << std::endl;
        printNodeResults(cancel_response.NodeResults());
    }

    printf("Example done...\n");

    client->Disconnect();
    return 0;
}
