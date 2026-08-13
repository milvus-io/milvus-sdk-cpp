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
}  // namespace
int
main() {
    auto client = milvus::MilvusClientV2::Create();
    // Connect authenticates with the configured Milvus endpoint for index administration.
    std::cout << "Calling Connect..." << std::endl;
    auto status = client->Connect(
        milvus::ConnectParam{Env("MILVUS_URI", "http://localhost:19530"), Env("MILVUS_TOKEN", "root:Milvus")});
    if (!Ok(status, "connect")) {
        return 1;
    }
    std::cout << "Connect succeeded." << std::endl;
    const std::string collection = "CPP_TUTORIAL_INDEX";
    std::cout << "Calling DropCollection for stale data..." << std::endl;
    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection));
    if (!Ok(status, "drop stale collection")) {
        return 1;
    }
    std::cout << "Stale collection cleanup completed." << std::endl;

    auto schema = std::make_shared<milvus::CollectionSchema>(collection, "Index tutorial", 1, false);
    schema->AddField({"id", milvus::DataType::INT64, "", true, false});
    schema->AddField(milvus::FieldSchema("category", milvus::DataType::VARCHAR).WithMaxLength(128));
    schema->AddField(milvus::FieldSchema("price", milvus::DataType::FLOAT));
    schema->AddField(milvus::FieldSchema("embedding", milvus::DataType::FLOAT_VECTOR).WithDimension(8));
    // CreateCollection creates the fields targeted by the indexes. No index is supplied here
    // because this tutorial demonstrates CreateIndex separately.
    std::cout << "Calling CreateCollection..." << std::endl;
    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection).WithCollectionSchema(schema));
    if (!Ok(status, "create collection")) {
        return 1;
    }
    std::cout << "CreateCollection succeeded." << std::endl;

    milvus::IndexDesc vector_index("embedding", "embedding_hnsw_idx", milvus::IndexType::HNSW,
                                   milvus::MetricType::COSINE);
    vector_index.AddExtraParam("M", "16");
    vector_index.AddExtraParam("efConstruction", "100");
    auto request = milvus::CreateIndexRequest()
                       .WithCollectionName(collection)
                       .AddIndex(std::move(vector_index))
                       .AddIndex(milvus::IndexDesc("category", "category_inverted_idx", milvus::IndexType::INVERTED))
                       .AddIndex(milvus::IndexDesc("price", "price_sort_idx", milvus::IndexType::STL_SORT));
    // CreateIndex builds all supplied index definitions. Each IndexDesc identifies a field,
    // index name and type, plus any metric or build parameters.
    std::cout << "Calling CreateIndex..." << std::endl;
    status = client->CreateIndex(request);
    if (!Ok(status, "create indexes")) {
        return 1;
    }
    std::cout << "CreateIndex succeeded." << std::endl;

    milvus::ListIndexesResponse listed;
    // ListIndexes returns every index defined on the selected collection.
    std::cout << "Calling ListIndexes..." << std::endl;
    status = client->ListIndexes(milvus::ListIndexesRequest().WithCollectionName(collection), listed);
    if (!Ok(status, "list indexes")) {
        return 1;
    }
    std::cout << "ListIndexes succeeded." << std::endl;
    for (const auto& index : listed.Descs()) {
        std::cout << index.IndexName() << " field=" << index.FieldName()
                  << " type=" << static_cast<int>(index.IndexType()) << std::endl;
    }

    milvus::DescribeIndexResponse described;
    // DescribeIndex returns detailed metadata for the selected index name.
    std::cout << "Calling DescribeIndex..." << std::endl;
    status = client->DescribeIndex(
        milvus::DescribeIndexRequest().WithCollectionName(collection).WithIndexName("embedding_hnsw_idx"), described);
    if (!Ok(status, "describe vector index")) {
        return 1;
    }
    std::cout << "DescribeIndex succeeded." << std::endl;
    std::cout << "Described vector indexes: " << described.Descs().size() << std::endl;

    // DropIndex removes only the named index; it does not delete its field or collection data.
    std::cout << "Calling DropIndex..." << std::endl;
    status =
        client->DropIndex(milvus::DropIndexRequest().WithCollectionName(collection).WithIndexName("price_sort_idx"));
    if (!Ok(status, "drop price index")) {
        return 1;
    }
    std::cout << "DropIndex succeeded." << std::endl;
    // DropCollection removes the tutorial collection and all remaining indexes.
    std::cout << "Calling DropCollection..." << std::endl;
    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection));
    if (!Ok(status, "drop collection")) {
        return 1;
    }
    std::cout << "DropCollection succeeded." << std::endl;
    return 0;
}
