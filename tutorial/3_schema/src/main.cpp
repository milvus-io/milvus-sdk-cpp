#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "milvus/MilvusClientV2.h"

namespace {

const char*
GetEnvOr(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value == nullptr ? fallback : value;
}

bool
CheckStatus(const milvus::Status& status, const std::string& operation) {
    if (!status.IsOk()) {
        std::cerr << "Failed to " << operation << ": " << status.Message() << std::endl;
        return false;
    }
    return true;
}

milvus::FieldSchema
PrimaryKey() {
    return {"id", milvus::DataType::INT64, "Primary key", true, false};
}

milvus::CollectionSchema
ScalarSchema() {
    milvus::CollectionSchema schema("scalar_schema_tutorial", "Scalar and container fields", 1, false);
    schema.AddField(PrimaryKey());
    schema.AddField(milvus::FieldSchema("bool_value", milvus::DataType::BOOL).WithNullable(true));
    schema.AddField(milvus::FieldSchema("int8_value", milvus::DataType::INT8).WithDefaultValue(0));
    schema.AddField(milvus::FieldSchema("int16_value", milvus::DataType::INT16));
    schema.AddField(milvus::FieldSchema("int32_value", milvus::DataType::INT32));
    schema.AddField(milvus::FieldSchema("int64_value", milvus::DataType::INT64));
    schema.AddField(milvus::FieldSchema("float_value", milvus::DataType::FLOAT));
    schema.AddField(milvus::FieldSchema("double_value", milvus::DataType::DOUBLE));
    schema.AddField(milvus::FieldSchema("varchar_value", milvus::DataType::VARCHAR).WithMaxLength(512));
    schema.AddField(milvus::FieldSchema("json_value", milvus::DataType::JSON));
    schema.AddField(milvus::FieldSchema("geometry_value", milvus::DataType::GEOMETRY));
    schema.AddField(milvus::FieldSchema("timestamp_value", milvus::DataType::TIMESTAMPTZ));
    schema.AddField(milvus::FieldSchema("int64_array", milvus::DataType::ARRAY)
                        .WithElementType(milvus::DataType::INT64)
                        .WithMaxCapacity(32));
    // Milvus collections require at least one vector field. This small vector keeps the collection
    // valid while the rest of the schema focuses on scalar and container types.
    schema.AddField(milvus::FieldSchema("vector", milvus::DataType::FLOAT_VECTOR).WithDimension(2));
    return schema;
}

milvus::CollectionSchema
VectorSchema() {
    milvus::CollectionSchema schema("vector_schema_tutorial", "Dense and binary vector fields", 1, false);
    schema.AddField(PrimaryKey());
    schema.AddField(milvus::FieldSchema("float_vector", milvus::DataType::FLOAT_VECTOR).WithDimension(8));
    schema.AddField(milvus::FieldSchema("binary_vector", milvus::DataType::BINARY_VECTOR).WithDimension(64));
    schema.AddField(milvus::FieldSchema("float16_vector", milvus::DataType::FLOAT16_VECTOR).WithDimension(8));
    schema.AddField(milvus::FieldSchema("bfloat16_vector", milvus::DataType::BFLOAT16_VECTOR).WithDimension(8));
    return schema;
}

milvus::CollectionSchema
StructSchema() {
    milvus::StructFieldSchema events("events", "An array of event records");
    events.WithMaxCapacity(16)
        .AddField(milvus::FieldSchema("label", milvus::DataType::VARCHAR).WithMaxLength(128))
        .AddField(milvus::FieldSchema("position", milvus::DataType::INT32))
        .AddField(milvus::FieldSchema("embedding", milvus::DataType::FLOAT_VECTOR).WithDimension(8));

    milvus::CollectionSchema schema("struct_schema_tutorial", "Sparse, Int8, and struct fields", 1, false);
    schema.AddField(PrimaryKey());
    schema.AddField(milvus::FieldSchema("sparse_vector", milvus::DataType::SPARSE_FLOAT_VECTOR));
    schema.AddField(milvus::FieldSchema("int8_vector", milvus::DataType::INT8_VECTOR).WithDimension(8));
    schema.AddStructField(std::move(events));
    return schema;
}

bool
CreateAndDescribe(milvus::MilvusClientV2Ptr& client, const std::string& name, milvus::CollectionSchema schema) {
    std::cout << "Calling DropCollection for stale collection " << name << "..." << std::endl;
    auto status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(name));
    if (!CheckStatus(status, "drop stale collection " + name)) {
        return false;
    }
    std::cout << "Stale collection cleanup completed for " << name << "." << std::endl;

    auto request = milvus::CreateCollectionRequest()
                       .WithCollectionName(name)
                       .WithDescription(schema.Description())
                       .WithCollectionSchema(std::make_shared<milvus::CollectionSchema>(std::move(schema)));

    // CreateCollection persists this complete schema. Splitting the tutorial across collections
    // keeps each schema within common vector-field limits while covering every supported type.
    std::cout << "Calling CreateCollection for " << name << "..." << std::endl;
    if (!CheckStatus(client->CreateCollection(request), "create collection " + name)) {
        return false;
    }
    std::cout << "CreateCollection succeeded for " << name << "." << std::endl;

    milvus::DescribeCollectionResponse response;

    // DescribeCollection reads the schema back from Milvus so the server representation of every
    // scalar, vector, array, and struct field can be inspected.
    std::cout << "Calling DescribeCollection for " << name << "..." << std::endl;
    if (!CheckStatus(client->DescribeCollection(milvus::DescribeCollectionRequest().WithCollectionName(name), response),
                     "describe collection " + name)) {
        return false;
    }
    std::cout << "DescribeCollection succeeded for " << name << "." << std::endl;

    std::cout << "\n" << name << ": " << response.Desc().Schema().Fields().size() << " fields" << std::endl;
    for (const auto& field : response.Desc().Schema().Fields()) {
        std::cout << "  " << field.Name() << " (type=" << static_cast<int>(field.FieldDataType()) << ")" << std::endl;
    }
    for (const auto& field : response.Desc().Schema().StructFields()) {
        std::cout << "  " << field.Name() << " (struct, " << field.Fields().size() << " sub-fields)" << std::endl;
    }
    return true;
}

}  // namespace

int
main() {
    auto client = milvus::MilvusClientV2::Create();
    milvus::ConnectParam connect_param{GetEnvOr("MILVUS_URI", "http://localhost:19530"),
                                       GetEnvOr("MILVUS_TOKEN", "root:Milvus")};

    // Connect authenticates all schema creation, description, and cleanup operations.
    std::cout << "Calling Connect..." << std::endl;
    auto status = client->Connect(connect_param);
    if (!CheckStatus(status, "connect")) {
        return 1;
    }
    std::cout << "Connect succeeded." << std::endl;

    const std::vector<std::pair<std::string, milvus::CollectionSchema>> tutorials = {
        {"CPP_TUTORIAL_SCHEMA_SCALAR", ScalarSchema()},
        {"CPP_TUTORIAL_SCHEMA_VECTOR", VectorSchema()},
        {"CPP_TUTORIAL_SCHEMA_STRUCT", StructSchema()},
    };

    for (const auto& tutorial : tutorials) {
        if (!CreateAndDescribe(client, tutorial.first, tutorial.second)) {
            return 1;
        }
    }

    for (const auto& tutorial : tutorials) {
        // DropCollection removes each temporary schema collection after it has been described.
        std::cout << "Calling DropCollection for " << tutorial.first << "..." << std::endl;
        status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(tutorial.first));
        if (!CheckStatus(status, "drop collection " + tutorial.first)) {
            return 1;
        }
        std::cout << "DropCollection succeeded for " << tutorial.first << "." << std::endl;
    }

    return 0;
}
