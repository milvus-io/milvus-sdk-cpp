#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

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
bool
Has(milvus::MilvusClientV2Ptr& client, const std::string& name) {
    milvus::HasCollectionResponse r;

    // HasCollection reports whether the named collection exists in the selected database.
    std::cout << "Calling HasCollection for " << name << "..." << std::endl;
    auto s = client->HasCollection(milvus::HasCollectionRequest().WithCollectionName(name), r);
    std::cout << "HasCollection completed for " << name << "." << std::endl;
    return s.IsOk() && r.Has();
}
}  // namespace
int
main() {
    auto client = milvus::MilvusClientV2::Create();

    // Connect creates one authenticated client that is reused for the complete collection
    // lifecycle demonstrated below.
    std::cout << "Calling Connect..." << std::endl;
    auto status = client->Connect(
        milvus::ConnectParam{Env("MILVUS_URI", "http://localhost:19530"), Env("MILVUS_TOKEN", "root:Milvus")});
    if (!Ok(status, "connect")) {
        return 1;
    }
    std::cout << "Connect succeeded." << std::endl;
    const std::string collection = "CPP_TUTORIAL_COLLECTION";
    const std::string renamed = collection + "_RENAMED";
    std::cout << "Calling DropCollection for stale collection " << collection << "..." << std::endl;
    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection));
    if (!Ok(status, "drop stale collection " + collection)) {
        return 1;
    }
    std::cout << "DropCollection completed for " << collection << "." << std::endl;
    std::cout << "Calling DropCollection for stale collection " << renamed << "..." << std::endl;
    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(renamed));
    if (!Ok(status, "drop stale collection " + renamed)) {
        return 1;
    }
    std::cout << "DropCollection completed for " << renamed << "." << std::endl;

    auto schema = std::make_shared<milvus::CollectionSchema>(collection, "Collection lifecycle tutorial", 1, false);
    schema->AddField({"id", milvus::DataType::INT64, "", true, false});
    schema->AddField(milvus::FieldSchema("title", milvus::DataType::VARCHAR).WithMaxLength(256));
    schema->AddField(milvus::FieldSchema("embedding", milvus::DataType::FLOAT_VECTOR).WithDimension(4));

    // CreateCollection persists the supplied schema and adds an AutoIndex so the collection can
    // be loaded later in this lifecycle tutorial. BOUNDED becomes the default read consistency.
    std::cout << "Calling CreateCollection..." << std::endl;
    status = client->CreateCollection(
        milvus::CreateCollectionRequest()
            .WithCollectionName(collection)
            .WithCollectionSchema(schema)
            .WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED)
            .AddIndex(milvus::IndexDesc("embedding", "", milvus::IndexType::AUTOINDEX, milvus::MetricType::COSINE)));
    if (!Ok(status, "create collection")) {
        return 1;
    }
    std::cout << "CreateCollection succeeded." << std::endl;

    milvus::HasCollectionResponse has_response;

    // HasCollection verifies that collection creation is visible in the selected database.
    std::cout << "Calling HasCollection..." << std::endl;
    status = client->HasCollection(milvus::HasCollectionRequest().WithCollectionName(collection), has_response);
    if (!Ok(status, "check collection")) {
        return 1;
    }
    std::cout << "HasCollection succeeded." << std::endl;
    std::cout << "Collection exists: " << has_response.Has() << std::endl;

    milvus::DescribeCollectionResponse described;

    // DescribeCollection returns collection metadata, properties, and the schema stored by Milvus.
    std::cout << "Calling DescribeCollection..." << std::endl;
    status = client->DescribeCollection(milvus::DescribeCollectionRequest().WithCollectionName(collection), described);
    if (!Ok(status, "describe collection")) {
        return 1;
    }
    std::cout << "DescribeCollection succeeded." << std::endl;
    std::cout << "Collection ID: " << described.Desc().ID() << std::endl;

    // AlterCollectionProperties adds or replaces property values. This property asks Milvus to
    // expire entities after 3,600 seconds.
    std::cout << "Calling AlterCollectionProperties..." << std::endl;
    status = client->AlterCollectionProperties(milvus::AlterCollectionPropertiesRequest()
                                                   .WithCollectionName(collection)
                                                   .AddProperty("collection.ttl.seconds", "3600"));
    if (!Ok(status, "alter collection properties")) {
        return 1;
    }
    std::cout << "AlterCollectionProperties succeeded." << std::endl;

    // DescribeCollection is called again to read back the updated property map.
    std::cout << "Calling DescribeCollection again..." << std::endl;
    status = client->DescribeCollection(milvus::DescribeCollectionRequest().WithCollectionName(collection), described);
    if (!Ok(status, "describe collection properties")) {
        return 1;
    }
    std::cout << "DescribeCollection succeeded." << std::endl;
    std::cout << "TTL: " << described.Desc().Properties().at("collection.ttl.seconds") << std::endl;

    // DropCollectionProperties removes only the selected property key and preserves the rest of
    // the collection configuration.
    std::cout << "Calling DropCollectionProperties..." << std::endl;
    status = client->DropCollectionProperties(milvus::DropCollectionPropertiesRequest()
                                                  .WithCollectionName(collection)
                                                  .AddPropertyKey("collection.ttl.seconds"));
    if (!Ok(status, "drop collection property")) {
        return 1;
    }
    std::cout << "DropCollectionProperties succeeded." << std::endl;

    // LoadCollection prepares the collection for reads. Sync mode waits for readiness and the
    // timeout prevents an unbounded wait.
    std::cout << "Calling LoadCollection..." << std::endl;
    status = client->LoadCollection(
        milvus::LoadCollectionRequest().WithCollectionName(collection).WithSync(true).WithTimeoutMs(60000));
    if (!Ok(status, "load collection")) {
        return 1;
    }
    std::cout << "LoadCollection succeeded." << std::endl;

    milvus::GetLoadStateResponse load_state;

    // GetLoadState returns both the current serving state and loading progress.
    std::cout << "Calling GetLoadState..." << std::endl;
    status = client->GetLoadState(milvus::GetLoadStateRequest().WithCollectionName(collection), load_state);
    if (!Ok(status, "get load state")) {
        return 1;
    }
    std::cout << "GetLoadState succeeded." << std::endl;
    std::cout << "Load state: " << static_cast<int>(load_state.State()) << ", progress=" << load_state.Progress() << "%"
              << std::endl;

    milvus::GetCollectionStatsResponse stats;

    // GetCollectionStats returns server statistics such as the collection row count.
    std::cout << "Calling GetCollectionStats..." << std::endl;
    status = client->GetCollectionStats(milvus::GetCollectionStatsRequest().WithCollectionName(collection), stats);
    if (!Ok(status, "get collection stats")) {
        return 1;
    }
    std::cout << "GetCollectionStats succeeded." << std::endl;
    std::cout << "Row count: " << stats.Stats().RowCount() << std::endl;

    // ReleaseCollection removes the collection from serving memory without deleting its data.
    std::cout << "Calling ReleaseCollection..." << std::endl;
    status = client->ReleaseCollection(milvus::ReleaseCollectionRequest().WithCollectionName(collection));
    if (!Ok(status, "release collection")) {
        return 1;
    }
    std::cout << "ReleaseCollection succeeded." << std::endl;

    // RenameCollection changes the collection name while preserving schema, data, and indexes.
    std::cout << "Calling RenameCollection..." << std::endl;
    status = client->RenameCollection(
        milvus::RenameCollectionRequest().WithCollectionName(collection).WithNewCollectionName(renamed));
    if (!Ok(status, "rename collection")) {
        return 1;
    }
    std::cout << "RenameCollection succeeded." << std::endl;
    std::cout << "Renamed collection exists: " << Has(client, renamed) << std::endl;

    // TruncateCollection deletes every entity while preserving the collection schema and indexes.
    std::cout << "Calling TruncateCollection..." << std::endl;
    status = client->TruncateCollection(milvus::TruncateCollectionRequest().WithCollectionName(renamed));
    if (!Ok(status, "truncate collection")) {
        return 1;
    }
    std::cout << "TruncateCollection succeeded." << std::endl;

    // DropCollection permanently removes the renamed tutorial collection.
    std::cout << "Calling DropCollection..." << std::endl;
    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(renamed));
    if (!Ok(status, "drop collection")) {
        return 1;
    }
    std::cout << "DropCollection succeeded." << std::endl;
    return 0;
}
