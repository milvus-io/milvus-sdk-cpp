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

#include <iostream>
#include <string>
#include <vector>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {

const std::string COLLECTION_NAME = "cpp_sdk_example_rls_collection";
const std::string DEMO_POLICY = "cpp_sdk_example_policy";
const std::string ALICE = "alice";
const std::string BOB = "bob";

void
DropRowPolicy(milvus::MilvusClientV2Ptr& client) {
    auto status = client->DropRowPolicy(
        milvus::DropRowPolicyRequest().WithCollectionName(COLLECTION_NAME).WithPolicyName(DEMO_POLICY));
    if (!status.IsOk()) {
        std::cout << "Drop row policy failed (ignored): " << status.Message() << std::endl;
    }
}

void
DropCollection(milvus::MilvusClientV2Ptr& client) {
    auto status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(COLLECTION_NAME));
    if (!status.IsOk()) {
        std::cout << "Drop collection failed (ignored): " << status.Message() << std::endl;
    }
}

milvus::EntityRow
Row(int64_t id, const std::string& team) {
    return {{"id", id}, {"vector", std::vector<float>{1.0f, 2.0f}}, {"team", team}};
}

void
CreateCollection(milvus::MilvusClientV2Ptr& client) {
    auto collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->AddField({"id", milvus::DataType::INT64, "", true, false});
    collection_schema->AddField(milvus::FieldSchema("vector", milvus::DataType::FLOAT_VECTOR).WithDimension(2));
    collection_schema->AddField(milvus::FieldSchema("team", milvus::DataType::VARCHAR).WithMaxLength(64));
    auto status = client->CreateCollection(milvus::CreateCollectionRequest()
                                               .WithCollectionName(COLLECTION_NAME)
                                               .WithCollectionSchema(collection_schema)
                                               .WithProperties({{"rls.enabled", "true"}}));
    util::CheckStatus("create collection: " + COLLECTION_NAME, status);
}

}  // namespace

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClientV2::Create();
    milvus::ConnectParam connect_param{"http://localhost:19530", "root:Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    // clean up resources created by a previous run
    DropRowPolicy(client);
    DropCollection(client);

    CreateCollection(client);

    // A policy expression may reference the current principal and its tags:
    //   $current_principal_tags['team']  ->  the 'team' tag of the request principal
    status = client->CreateRowPolicy(milvus::CreateRowPolicyRequest()
                                         .WithCollectionName(COLLECTION_NAME)
                                         .WithPolicyName(DEMO_POLICY)
                                         .WithPolicyType(milvus::RowPolicyType::PERMISSIVE)
                                         .WithActions({milvus::RowPolicyAction::SEARCH, milvus::RowPolicyAction::QUERY,
                                                       milvus::RowPolicyAction::INSERT})
                                         .WithUsingExpr("team == $current_principal_tags['team']")
                                         .WithCheckExpr("team == $current_principal_tags['team']")
                                         .WithDescription("A principal only sees or writes rows of its own team"));
    util::CheckStatus("create row policy: " + DEMO_POLICY, status);

    {
        milvus::ListRowPoliciesResponse response;
        status =
            client->ListRowPolicies(milvus::ListRowPoliciesRequest().WithCollectionName(COLLECTION_NAME), response);
        util::CheckStatus("list row policies", status);
        std::cout << "row policies after create:" << std::endl;
        for (const auto& policy : response.Policies()) {
            std::cout << "{name:" << policy.PolicyName() << ", type:" << static_cast<int>(policy.PolicyType())
                      << ", actions:" << policy.Actions().size() << ", using_expr:" << policy.UsingExpr()
                      << ", check_expr:" << policy.CheckExpr() << ", description:" << policy.Description() << "}"
                      << std::endl;
        }
    }

    status = client->UpdateRowPolicy(
        milvus::UpdateRowPolicyRequest()
            .WithCollectionName(COLLECTION_NAME)
            .WithPolicyName(DEMO_POLICY)
            .WithPolicyType(milvus::RowPolicyType::PERMISSIVE)
            .WithActions({milvus::RowPolicyAction::SEARCH, milvus::RowPolicyAction::QUERY,
                          milvus::RowPolicyAction::INSERT, milvus::RowPolicyAction::DELETE_ROWS})
            .WithUsingExpr("team == $current_principal_tags['team']")
            .WithCheckExpr("team == $current_principal_tags['team']")
            .WithDescription("Updated: a principal sees or writes only rows of its own team"));
    util::CheckStatus("update row policy: " + DEMO_POLICY, status);

    status = client->SetRLSPrincipalTags(milvus::SetRLSPrincipalTagsRequest()
                                             .WithCollectionName(COLLECTION_NAME)
                                             .WithPrincipalName(ALICE)
                                             .WithTags({{"team", "engineering"}, {"level", 3}}));
    util::CheckStatus("set principal tags for alice", status);

    status = client->SetRLSPrincipalTags(milvus::SetRLSPrincipalTagsRequest()
                                             .WithCollectionName(COLLECTION_NAME)
                                             .WithPrincipalName(BOB)
                                             .WithTags({{"team", "marketing"}}));
    util::CheckStatus("set principal tags for bob", status);

    {
        milvus::GetRLSPrincipalTagsRequest request;
        request.WithCollectionName(COLLECTION_NAME).WithPrincipalName(ALICE);
        milvus::GetRLSPrincipalTagsResponse response;
        status = client->GetRLSPrincipalTags(request, response);
        util::CheckStatus("get principal tags of alice", status);
        std::cout << "alice tags: " << response.Tags().dump() << std::endl;
    }

    {
        milvus::ListRLSPrincipalsResponse response;
        status =
            client->ListRLSPrincipals(milvus::ListRLSPrincipalsRequest().WithCollectionName(COLLECTION_NAME), response);
        util::CheckStatus("list principals with tags", status);
        std::cout << "principals with tags: ";
        util::PrintList(response.PrincipalNames());
    }

    status = client->DeleteRLSPrincipalTags(milvus::DeleteRLSPrincipalTagsRequest()
                                                .WithCollectionName(COLLECTION_NAME)
                                                .WithPrincipalName(ALICE)
                                                .WithTagKeys({"level"}));
    util::CheckStatus("delete tag 'level' of alice", status);

    // ---- enforcement: writes are checked against the policy ----
    milvus::InsertResponse insert_response;
    status = client->Insert(milvus::InsertRequest()
                                .WithCollectionName(COLLECTION_NAME)
                                .WithRowsData({Row(1, "engineering")})
                                .WithRLSPrincipal(ALICE),
                            insert_response);
    util::CheckStatus("insert engineering row as alice", status);

    status = client->Insert(milvus::InsertRequest()
                                .WithCollectionName(COLLECTION_NAME)
                                .WithRowsData({Row(2, "marketing")})
                                .WithRLSPrincipal(ALICE),
                            insert_response);
    if (status.IsOk()) {
        std::cout << "UNEXPECTED: insert marketing row as alice succeeded" << std::endl;
    } else {
        std::cout << "Insert marketing row as alice rejected by RLS check expression: " << status.Message()
                  << std::endl;
    }

    status = client->Insert(milvus::InsertRequest()
                                .WithCollectionName(COLLECTION_NAME)
                                .WithRowsData({Row(3, "marketing")})
                                .WithRLSPrincipal(BOB),
                            insert_response);
    util::CheckStatus("insert marketing row as bob", status);

    // load the collection so query can run
    milvus::IndexParam index_vector("vector", "", milvus::IndexType::FLAT, milvus::MetricType::L2);
    status = client->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(COLLECTION_NAME).AddIndexParam(std::move(index_vector)));
    util::CheckStatus("create index on vector field", status);

    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(COLLECTION_NAME));
    util::CheckStatus("load collection: " + COLLECTION_NAME, status);

    // ---- enforcement: reads are filtered by the principal's tags ----
    milvus::QueryResponse query_response;
    status = client->Query(
        milvus::QueryRequest().WithCollectionName(COLLECTION_NAME).WithFilter("id > 0").WithRLSPrincipal(ALICE),
        query_response);
    util::CheckStatus("query rows as alice", status);
    std::cout << "rows visible to alice: " << query_response.Results().GetRowCount() << std::endl;

    status = client->Query(
        milvus::QueryRequest().WithCollectionName(COLLECTION_NAME).WithFilter("id > 0").WithRLSPrincipal(BOB),
        query_response);
    util::CheckStatus("query rows as bob", status);
    std::cout << "rows visible to bob: " << query_response.Results().GetRowCount() << std::endl;

    status =
        client->Query(milvus::QueryRequest().WithCollectionName(COLLECTION_NAME).WithFilter("id > 0").WithSkipRLS(true),
                      query_response);
    util::CheckStatus("query rows with skipRls", status);
    std::cout << "rows visible with skipRls: " << query_response.Results().GetRowCount() << std::endl;

    status =
        client->Query(milvus::QueryRequest().WithCollectionName(COLLECTION_NAME).WithFilter("id > 0"), query_response);
    if (status.IsOk()) {
        std::cout << "UNEXPECTED: query without rls_principal succeeded" << std::endl;
    } else {
        std::cout << "Query without rls_principal denied: " << status.Message() << std::endl;
    }

    status = client->DropRowPolicy(
        milvus::DropRowPolicyRequest().WithCollectionName(COLLECTION_NAME).WithPolicyName(DEMO_POLICY));
    util::CheckStatus("drop row policy: " + DEMO_POLICY, status);

    // clean up resources created by this run
    DropCollection(client);

    client->Disconnect();
    return 0;
}
