#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "milvus/MilvusClientV2.h"
namespace {
const char*
Env(const char* n, const char* d) {
    const char* v = std::getenv(n);
    return v ? v : d;
}
bool
Ok(const milvus::Status& s, const std::string& op) {
    if (s.IsOk()) {
        return true;
    }
    std::cerr << "Failed to " << op << ": " << s.Message() << std::endl;
    return false;
}
}  // namespace
int
main() {
    auto client = milvus::MilvusClientV2::Create();

    // Connect establishes the client channel. The URI selects the Milvus endpoint and the token
    // supplies an API key or username/password credential for all following calls.
    std::cout << "Calling Connect..." << std::endl;
    auto status = client->Connect(
        milvus::ConnectParam{Env("MILVUS_URI", "http://localhost:19530"), Env("MILVUS_TOKEN", "root:Milvus")});
    if (!Ok(status, "connect")) {
        return 1;
    }
    std::cout << "Connect succeeded." << std::endl;
    const std::string collection = "CPP_TUTORIAL_QUICKSTART";
    std::cout << "Calling DropCollection for stale data..." << std::endl;
    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection));
    if (!Ok(status, "drop stale collection")) {
        return 1;
    }
    std::cout << "Stale collection cleanup completed." << std::endl;

    auto schema = std::make_shared<milvus::CollectionSchema>(collection, "Quickstart", 1, false);
    schema->AddField({"id", milvus::DataType::INT64, "", true, false});
    schema->AddField(milvus::FieldSchema("text", milvus::DataType::VARCHAR).WithMaxLength(256));
    schema->AddField(milvus::FieldSchema("embedding", milvus::DataType::FLOAT_VECTOR).WithDimension(4));

    // CreateCollection persists the schema under collection_name. The attached AutoIndex makes
    // the embedding field searchable with cosine similarity.
    std::cout << "Calling CreateCollection..." << std::endl;
    status = client->CreateCollection(
        milvus::CreateCollectionRequest()
            .WithCollectionName(collection)
            .WithCollectionSchema(schema)
            .AddIndex(milvus::IndexDesc("embedding", "", milvus::IndexType::AUTOINDEX, milvus::MetricType::COSINE)));
    if (!Ok(status, "create collection")) {
        return 1;
    }
    std::cout << "CreateCollection succeeded." << std::endl;
    milvus::EntityRows rows = {{{"id", 1}, {"text", "hello"}, {"embedding", std::vector<float>{.1f, .2f, .3f, .4f}}},
                               {{"id", 2}, {"text", "milvus"}, {"embedding", std::vector<float>{.2f, .3f, .4f, .5f}}}};
    milvus::InsertResponse inserted;

    // Insert writes row-oriented entities to the selected collection. Every row must match the
    // declared field names and types; the response reports the accepted row count and IDs.
    std::cout << "Calling Insert..." << std::endl;
    status =
        client->Insert(milvus::InsertRequest().WithCollectionName(collection).WithRowsData(std::move(rows)), inserted);
    if (!Ok(status, "insert")) {
        return 1;
    }
    std::cout << "Insert succeeded: " << inserted.Results().InsertCount() << " rows." << std::endl;

    // LoadCollection prepares collection data and indexes for query serving. Sync mode waits for
    // readiness, while timeout_ms bounds the wait to 60 seconds.
    std::cout << "Calling LoadCollection..." << std::endl;
    status = client->LoadCollection(
        milvus::LoadCollectionRequest().WithCollectionName(collection).WithSync(true).WithTimeoutMs(60000));
    if (!Ok(status, "load collection")) {
        return 1;
    }
    std::cout << "LoadCollection succeeded." << std::endl;

    milvus::SearchResponse response;

    // Search performs nearest-neighbor lookup on embedding. The query vector is compared with
    // cosine similarity, output_fields returns text, limit caps the number of matches, and strong
    // consistency makes the preceding insert visible.
    std::cout << "Calling Search..." << std::endl;
    status = client->Search(milvus::SearchRequest()
                                .WithCollectionName(collection)
                                .WithAnnsField("embedding")
                                .AddOutputField("text")
                                .WithLimit(2)
                                .AddFloatVector({.1f, .2f, .3f, .4f})
                                .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG),
                            response);
    if (!Ok(status, "search")) {
        return 1;
    }
    std::cout << "Search succeeded." << std::endl;
    std::cout << "Search result sets: " << response.Results().Results().size() << std::endl;

    // ReleaseCollection removes the collection from serving memory without deleting data.
    std::cout << "Calling ReleaseCollection..." << std::endl;
    status = client->ReleaseCollection(milvus::ReleaseCollectionRequest().WithCollectionName(collection));
    if (!Ok(status, "release collection")) {
        return 1;
    }
    std::cout << "ReleaseCollection succeeded." << std::endl;

    // DropCollection permanently removes the tutorial collection, its data, and its indexes.
    std::cout << "Calling DropCollection..." << std::endl;
    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection));
    if (!Ok(status, "drop collection")) {
        return 1;
    }
    std::cout << "DropCollection succeeded." << std::endl;
    return 0;
}
