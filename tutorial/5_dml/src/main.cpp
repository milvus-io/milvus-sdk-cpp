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
    // Connect authenticates with the configured Milvus endpoint for all following DML requests.
    std::cout << "Calling Connect..." << std::endl;
    auto status = client->Connect(
        milvus::ConnectParam{Env("MILVUS_URI", "http://localhost:19530"), Env("MILVUS_TOKEN", "root:Milvus")});
    if (!Ok(status, "connect")) {
        return 1;
    }
    std::cout << "Connect succeeded." << std::endl;
    const std::string collection = "CPP_TUTORIAL_DML";
    std::cout << "Calling DropCollection for stale data..." << std::endl;
    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection));
    if (!Ok(status, "drop stale collection")) {
        return 1;
    }
    std::cout << "Stale collection cleanup completed." << std::endl;
    auto schema = std::make_shared<milvus::CollectionSchema>(collection, "DML tutorial", 1, false);
    schema->AddField({"id", milvus::DataType::INT64, "", true, false});
    schema->AddField(milvus::FieldSchema("title", milvus::DataType::VARCHAR).WithMaxLength(256));
    schema->AddField(milvus::FieldSchema("price", milvus::DataType::FLOAT));
    schema->AddField(milvus::FieldSchema("embedding", milvus::DataType::FLOAT_VECTOR).WithDimension(4));
    // CreateCollection creates the DML target. The schema defines valid input fields, and the
    // supplied index makes the vector field searchable.
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
    // LoadCollection prepares the collection for serving. WithSync waits until loading finishes,
    // bounded by the configured timeout.
    std::cout << "Calling LoadCollection..." << std::endl;
    status = client->LoadCollection(
        milvus::LoadCollectionRequest().WithCollectionName(collection).WithSync(true).WithTimeoutMs(60000));
    if (!Ok(status, "load collection")) {
        return 1;
    }
    std::cout << "LoadCollection succeeded." << std::endl;

    milvus::EntityRows rows = {{{"id", 1},
                                {"title", "Rust in Action"},
                                {"price", 35.0},
                                {"embedding", std::vector<float>{.1f, .2f, .3f, .4f}}},
                               {{"id", 2},
                                {"title", "Vector Search"},
                                {"price", 25.0},
                                {"embedding", std::vector<float>{.2f, .3f, .4f, .5f}}}};
    milvus::InsertResponse insert_response;
    // Insert with RowsData writes row-oriented entities. Every row must follow the collection
    // schema, and the response reports how many entities were accepted.
    std::cout << "Calling Insert with row data..." << std::endl;
    status = client->Insert(milvus::InsertRequest().WithCollectionName(collection).WithRowsData(std::move(rows)),
                            insert_response);
    if (!Ok(status, "insert rows")) {
        return 1;
    }
    std::cout << "Insert with row data succeeded." << std::endl;
    std::cout << "Inserted rows: " << insert_response.Results().InsertCount() << std::endl;

    auto ids = std::vector<int64_t>{3, 4};
    auto titles = std::vector<std::string>{"Milvus Guide", "Database Systems"};
    auto prices = std::vector<float>{30.0f, 40.0f};
    auto vectors = std::vector<std::vector<float>>{{.3f, .4f, .5f, .6f}, {.4f, .5f, .6f, .7f}};
    std::vector<milvus::FieldDataPtr> columns = {std::make_shared<milvus::Int64FieldData>("id", ids),
                                                 std::make_shared<milvus::VarCharFieldData>("title", titles),
                                                 std::make_shared<milvus::FloatFieldData>("price", prices),
                                                 std::make_shared<milvus::FloatVecFieldData>("embedding", vectors)};
    // Insert also accepts column-oriented FieldData. Each column names a schema field, and all
    // columns must contain the same number of values.
    std::cout << "Calling Insert with column data..." << std::endl;
    status = client->Insert(milvus::InsertRequest().WithCollectionName(collection).WithColumnsData(std::move(columns)),
                            insert_response);
    if (!Ok(status, "insert columns")) {
        return 1;
    }
    std::cout << "Insert with column data succeeded." << std::endl;
    std::cout << "Inserted column rows: " << insert_response.Results().InsertCount() << std::endl;

    milvus::EntityRow upsert_row = {{"id", 2},
                                    {"title", "Practical Vector Search"},
                                    {"price", 27.5},
                                    {"embedding", std::vector<float>{.25f, .35f, .45f, .55f}}};
    milvus::UpsertResponse upsert_response;
    // Upsert replaces an existing entity or inserts it when the primary key is absent. A full
    // upsert supplies values for every required field.
    std::cout << "Calling Upsert for a full row..." << std::endl;
    status = client->Upsert(milvus::UpsertRequest().WithCollectionName(collection).AddRowData(std::move(upsert_row)),
                            upsert_response);
    if (!Ok(status, "upsert row")) {
        return 1;
    }
    std::cout << "Full Upsert succeeded." << std::endl;
    std::cout << "Upserted rows: " << upsert_response.Results().UpsertCount() << std::endl;

    milvus::EntityRow partial_row = {{"id", 3}, {"title", "The Milvus Guide"}};
    // A partial upsert changes only the supplied non-primary fields. The primary key still
    // identifies which entity to update.
    std::cout << "Calling Upsert for a partial update..." << std::endl;
    status = client->Upsert(milvus::UpsertRequest()
                                .WithCollectionName(collection)
                                .WithPartialUpdate(true)
                                .AddRowData(std::move(partial_row)),
                            upsert_response);
    if (!Ok(status, "partial upsert row")) {
        return 1;
    }
    std::cout << "Partial Upsert succeeded." << std::endl;

    milvus::DeleteResponse delete_response;
    // Delete with IDs removes entities by primary key from the selected collection.
    std::cout << "Calling Delete with primary-key IDs..." << std::endl;
    status = client->Delete(milvus::DeleteRequest().WithCollectionName(collection).WithIDs(std::vector<int64_t>{1}),
                            delete_response);
    if (!Ok(status, "delete by IDs")) {
        return 1;
    }
    std::cout << "Delete by IDs succeeded." << std::endl;
    // Delete with a filter removes all entities matching the Milvus boolean expression.
    std::cout << "Calling Delete with a filter..." << std::endl;
    status =
        client->Delete(milvus::DeleteRequest().WithCollectionName(collection).WithFilter("id >= 4"), delete_response);
    if (!Ok(status, "delete by filter")) {
        return 1;
    }
    std::cout << "Delete by filter succeeded." << std::endl;
    std::cout << "Deleted by filter: " << delete_response.Results().DeleteCount() << std::endl;

    // ReleaseCollection removes the tutorial collection from serving memory before deletion.
    std::cout << "Calling ReleaseCollection..." << std::endl;
    status = client->ReleaseCollection(milvus::ReleaseCollectionRequest().WithCollectionName(collection));
    if (!Ok(status, "release collection")) {
        return 1;
    }
    std::cout << "ReleaseCollection succeeded." << std::endl;
    // DropCollection permanently removes the tutorial collection and its remaining data.
    std::cout << "Calling DropCollection..." << std::endl;
    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection));
    if (!Ok(status, "drop collection")) {
        return 1;
    }
    std::cout << "DropCollection succeeded." << std::endl;
    return 0;
}
