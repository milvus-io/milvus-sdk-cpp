#include <cstdlib>
#include <iostream>
#include <map>
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
std::vector<float>
Dense(int id) {
    float b = id / 10.0f;
    return {b, b + .1f, b + .2f, b + .3f};
}
std::map<uint32_t, float>
Sparse(int id) {
    return {{static_cast<uint32_t>(id % 8), 1.0f}, {static_cast<uint32_t>((id + 3) % 8), .5f}};
}
nlohmann::json
SparseJson(int id) {
    nlohmann::json value = nlohmann::json::object();
    for (const auto& item : Sparse(id)) {
        value[std::to_string(item.first)] = item.second;
    }
    return value;
}
bool
PrintRows(const milvus::QueryResults& results) {
    milvus::EntityRows rows;
    auto status = results.OutputRows(rows);
    if (!Ok(status, "convert query results to rows")) {
        return false;
    }
    for (const auto& row : rows) {
        std::cout << "  " << row << std::endl;
    }
    return true;
}
bool
PrintSearch(const milvus::SearchResults& results) {
    for (const auto& result : results.Results()) {
        milvus::EntityRows rows;
        auto status = result.OutputRows(rows);
        if (!Ok(status, "convert search results to rows")) {
            return false;
        }
        for (const auto& row : rows) {
            std::cout << "  " << row << std::endl;
        }
    }
    return true;
}
}  // namespace
int
main() {
    auto client = milvus::MilvusClientV2::Create();
    // Connect authenticates with the configured endpoint for all query and search operations.
    std::cout << "Calling Connect..." << std::endl;
    auto status = client->Connect(
        milvus::ConnectParam{Env("MILVUS_URI", "http://localhost:19530"), Env("MILVUS_TOKEN", "root:Milvus")});
    if (!Ok(status, "connect")) {
        return 1;
    }
    std::cout << "Connect succeeded." << std::endl;
    const std::string collection = "CPP_TUTORIAL_DQL";
    std::cout << "Calling DropCollection for stale data..." << std::endl;
    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection));
    if (!Ok(status, "drop stale collection")) {
        return 1;
    }
    std::cout << "Stale collection cleanup completed." << std::endl;

    auto schema = std::make_shared<milvus::CollectionSchema>(collection, "DQL tutorial", 1, false);
    schema->AddField({"id", milvus::DataType::INT64, "", true, false});
    schema->AddField(milvus::FieldSchema("title", milvus::DataType::VARCHAR).WithMaxLength(256));
    schema->AddField(milvus::FieldSchema("category", milvus::DataType::INT32));
    schema->AddField(milvus::FieldSchema("dense", milvus::DataType::FLOAT_VECTOR).WithDimension(4));
    schema->AddField(milvus::FieldSchema("sparse", milvus::DataType::SPARSE_FLOAT_VECTOR));
    // CreateCollection creates the DQL target with indexes for both dense and sparse searches.
    std::cout << "Calling CreateCollection..." << std::endl;
    status = client->CreateCollection(
        milvus::CreateCollectionRequest()
            .WithCollectionName(collection)
            .WithCollectionSchema(schema)
            .AddIndex(milvus::IndexDesc("dense", "", milvus::IndexType::AUTOINDEX, milvus::MetricType::COSINE))
            .AddIndex(
                milvus::IndexDesc("sparse", "", milvus::IndexType::SPARSE_INVERTED_INDEX, milvus::MetricType::IP)));
    if (!Ok(status, "create collection")) {
        return 1;
    }
    std::cout << "CreateCollection succeeded." << std::endl;
    // LoadCollection prepares the collection for query and search. WithSync waits until serving
    // is ready, bounded by the configured timeout.
    std::cout << "Calling LoadCollection..." << std::endl;
    status = client->LoadCollection(
        milvus::LoadCollectionRequest().WithCollectionName(collection).WithSync(true).WithTimeoutMs(60000));
    if (!Ok(status, "load collection")) {
        return 1;
    }
    std::cout << "LoadCollection succeeded." << std::endl;

    milvus::EntityRows rows;
    for (int id = 0; id < 12; ++id) {
        rows.push_back({{"id", id},
                        {"title", "document_" + std::to_string(id)},
                        {"category", id % 3},
                        {"dense", Dense(id)},
                        {"sparse", SparseJson(id)}});
    }
    milvus::InsertResponse inserted;
    // Insert writes the tutorial entities that the following DQL operations will read.
    std::cout << "Calling Insert..." << std::endl;
    status =
        client->Insert(milvus::InsertRequest().WithCollectionName(collection).WithRowsData(std::move(rows)), inserted);
    if (!Ok(status, "insert tutorial data")) {
        return 1;
    }
    std::cout << "Insert succeeded." << std::endl;
    std::cout << "Inserted rows: " << inserted.Results().InsertCount() << std::endl;

    milvus::QueryResponse query_response;
    // Query performs scalar retrieval. The filter selects rows, output fields select returned
    // columns, limit caps the results, and strong consistency exposes the preceding insert.
    std::cout << "Calling Query..." << std::endl;
    status = client->Query(milvus::QueryRequest()
                               .WithCollectionName(collection)
                               .WithFilter("category == 1")
                               .AddOutputField("id")
                               .AddOutputField("title")
                               .AddOutputField("category")
                               .WithLimit(5)
                               .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG),
                           query_response);
    if (!Ok(status, "query")) {
        return 1;
    }
    std::cout << "Query succeeded." << std::endl;
    std::cout << "\nQuery results:" << std::endl;
    if (!PrintRows(query_response.Results())) {
        return 1;
    }

    milvus::SearchResponse search_response;
    // Search performs nearest-neighbor search on the dense field. The query vector supplies the
    // target embedding, the filter constrains candidates, and limit caps matches.
    std::cout << "Calling Search..." << std::endl;
    status = client->Search(milvus::SearchRequest()
                                .WithCollectionName(collection)
                                .WithAnnsField("dense")
                                .WithFilter("category >= 0")
                                .AddOutputField("title")
                                .AddOutputField("category")
                                .WithLimit(3)
                                .AddFloatVector(Dense(2))
                                .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG),
                            search_response);
    if (!Ok(status, "search")) {
        return 1;
    }
    std::cout << "Search succeeded." << std::endl;
    std::cout << "\nSearch results:" << std::endl;
    if (!PrintSearch(search_response.Results())) {
        return 1;
    }

    auto dense = milvus::SubSearchRequest().WithAnnsField("dense").WithLimit(6).AddFloatVector(Dense(2));
    auto sparse = milvus::SubSearchRequest().WithAnnsField("sparse").WithLimit(6).AddSparseVector(Sparse(2));
    auto rerank = std::make_shared<milvus::WeightedRerank>(std::vector<float>{.7f, .3f});
    // HybridSearch combines dense and sparse sub-searches. The weights control their relative
    // contribution, output fields select metadata, and limit caps reranked matches.
    std::cout << "Calling HybridSearch..." << std::endl;
    status = client->HybridSearch(milvus::HybridSearchRequest()
                                      .WithCollectionName(collection)
                                      .WithLimit(4)
                                      .AddSubRequest(std::make_shared<milvus::SubSearchRequest>(std::move(dense)))
                                      .AddSubRequest(std::make_shared<milvus::SubSearchRequest>(std::move(sparse)))
                                      .WithRerank(rerank)
                                      .AddOutputField("title")
                                      .AddOutputField("category")
                                      .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG),
                                  search_response);
    if (!Ok(status, "hybrid search")) {
        return 1;
    }
    std::cout << "HybridSearch succeeded." << std::endl;
    std::cout << "\nHybrid search results:" << std::endl;
    if (!PrintSearch(search_response.Results())) {
        return 1;
    }

    milvus::QueryIteratorRequest query_iterator_request;
    query_iterator_request.SetCollectionName(collection);
    query_iterator_request.SetFilter("id >= 0");
    query_iterator_request.AddOutputField("id");
    query_iterator_request.AddOutputField("title");
    query_iterator_request.SetBatchSize(3);
    query_iterator_request.SetLimit(7);
    milvus::QueryIteratorPtr query_iterator;
    // QueryIterator paginates a query. BatchSize controls rows per page, and Limit controls the
    // total number of rows returned across all pages.
    std::cout << "Calling QueryIterator..." << std::endl;
    status = client->QueryIterator(query_iterator_request, query_iterator);
    if (!Ok(status, "create query iterator")) {
        return 1;
    }
    std::cout << "QueryIterator succeeded." << std::endl;
    std::cout << "\nQuery iterator:" << std::endl;
    while (true) {
        milvus::QueryResults page;
        std::cout << "Calling QueryIterator::Next..." << std::endl;
        status = query_iterator->Next(page);
        if (!Ok(status, "query iterator next")) {
            return 1;
        }
        std::cout << "QueryIterator::Next succeeded with " << page.GetRowCount() << " rows." << std::endl;
        if (page.GetRowCount() == 0) {
            break;
        }
        if (!PrintRows(page)) {
            return 1;
        }
    }

    milvus::SearchIteratorRequest search_iterator_request;
    search_iterator_request.SetCollectionName(collection);
    search_iterator_request.SetAnnsField("dense");
    search_iterator_request.AddFloatVector(Dense(2));
    search_iterator_request.AddOutputField("title");
    search_iterator_request.SetBatchSize(3);
    search_iterator_request.SetLimit(7);
    milvus::SearchIteratorPtr search_iterator;
    // SearchIterator paginates a vector search while preserving one search session. BatchSize
    // controls page size, and Limit controls the total matches delivered.
    std::cout << "Calling SearchIterator..." << std::endl;
    status = client->SearchIterator(search_iterator_request, search_iterator);
    if (!Ok(status, "create search iterator")) {
        return 1;
    }
    std::cout << "SearchIterator succeeded." << std::endl;
    std::cout << "\nSearch iterator:" << std::endl;
    while (true) {
        milvus::SingleResult page;
        std::cout << "Calling SearchIterator::Next..." << std::endl;
        status = search_iterator->Next(page);
        if (!Ok(status, "search iterator next")) {
            return 1;
        }
        std::cout << "SearchIterator::Next succeeded with " << page.GetRowCount() << " rows." << std::endl;
        if (page.GetRowCount() == 0) {
            break;
        }
        milvus::EntityRows output;
        status = page.OutputRows(output);
        if (!Ok(status, "convert search iterator page to rows")) {
            return 1;
        }
        for (const auto& row : output) {
            std::cout << "  " << row << std::endl;
        }
    }

    // ReleaseCollection removes the tutorial collection from serving memory before deletion.
    std::cout << "Calling ReleaseCollection..." << std::endl;
    status = client->ReleaseCollection(milvus::ReleaseCollectionRequest().WithCollectionName(collection));
    if (!Ok(status, "release collection")) {
        return 1;
    }
    std::cout << "ReleaseCollection succeeded." << std::endl;
    // DropCollection permanently removes the tutorial collection and its indexes.
    std::cout << "Calling DropCollection..." << std::endl;
    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection));
    if (!Ok(status, "drop collection")) {
        return 1;
    }
    std::cout << "DropCollection succeeded." << std::endl;
    return 0;
}
