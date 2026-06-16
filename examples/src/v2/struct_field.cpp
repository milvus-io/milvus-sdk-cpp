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

#include <functional>
#include <iostream>
#include <string>
#include <thread>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {

const char* const collection_name = "CPP_V2_STRUCT";
const char* const field_id = "id";
const char* const field_name = "film_name";
const char* const field_struct = "clips";
const char* const field_struct_int32 = "frame_number";
const char* const field_struct_varchar = "clip_desc";
const char* const field_struct_vector = "clip_float_embedding";
const char* const field_struct_binary_vector = "clip_binary_embedding";
const char* const field_struct_fp16_vector = "clip_float16_embedding";
const char* const field_struct_bf16_vector = "clip_bfloat16_embedding";
const char* const field_struct_int8_vector = "clip_int8_embedding";
const char* const field_simplify_struct = "simplify_clips";
const char* const field_extra_struct = "metadata";
const char* const field_extra_struct_rating = "rating";
const char* const field_extra_struct_tag = "tag";
const int64_t extra_row_id = 5000;

const uint32_t float_dimension = 16;
const uint32_t binary_dimension = 16;
const uint32_t fp16_dimension = 16;
const uint32_t int8_dimension = 16;
const uint32_t simplify_dimension = 32;
const int64_t struct_capacity = 100;

std::string
combineStructName(const std::string& struct_field, const std::string& sub_field) {
    return struct_field + "[" + sub_field + "]";
}

void
createCollection(milvus::MilvusClientV2Ptr& client) {
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->SetEnableDynamicField(false);
    collection_schema->AddField(milvus::FieldSchema(field_id, milvus::DataType::INT64, "id", true, false));
    collection_schema->AddField(milvus::FieldSchema(field_name, milvus::DataType::VARCHAR).WithMaxLength(1024));

    milvus::StructFieldSchema struct_schema =
        milvus::StructFieldSchema()
            .WithName(field_struct)
            .WithMaxCapacity(struct_capacity)
            .AddField(milvus::FieldSchema(field_struct_int32, milvus::DataType::INT32))
            .AddField(milvus::FieldSchema(field_struct_varchar, milvus::DataType::VARCHAR).WithMaxLength(1024))
            .AddField(
                milvus::FieldSchema(field_struct_vector, milvus::DataType::FLOAT_VECTOR).WithDimension(float_dimension))
            .AddField(milvus::FieldSchema(field_struct_binary_vector, milvus::DataType::BINARY_VECTOR)
                          .WithDimension(binary_dimension))
            .AddField(milvus::FieldSchema(field_struct_fp16_vector, milvus::DataType::FLOAT16_VECTOR)
                          .WithDimension(fp16_dimension))
            .AddField(milvus::FieldSchema(field_struct_bf16_vector, milvus::DataType::BFLOAT16_VECTOR)
                          .WithDimension(fp16_dimension))
            .AddField(milvus::FieldSchema(field_struct_int8_vector, milvus::DataType::INT8_VECTOR)
                          .WithDimension(int8_dimension));
    collection_schema->AddStructField(std::move(struct_schema));

    milvus::StructFieldSchema simplify_struct_schema =
        milvus::StructFieldSchema()
            .WithName(field_simplify_struct)
            .WithMaxCapacity(struct_capacity)
            .AddField(milvus::FieldSchema(field_struct_vector, milvus::DataType::FLOAT_VECTOR)
                          .WithDimension(simplify_dimension));
    collection_schema->AddStructField(std::move(simplify_struct_schema));

    auto status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + std::string(collection_name), status);

    milvus::IndexDesc index_struct_float(combineStructName(field_struct, field_struct_vector), "index_float",
                                         milvus::IndexType::HNSW, milvus::MetricType::MAX_SIM_IP);
    milvus::IndexDesc index_struct_binary(combineStructName(field_struct, field_struct_binary_vector), "index_binary",
                                          milvus::IndexType::HNSW, milvus::MetricType::MAX_SIM_HAMMING);
    milvus::IndexDesc index_struct_fp16(combineStructName(field_struct, field_struct_fp16_vector), "index_float16",
                                        milvus::IndexType::IVF_FLAT, milvus::MetricType::MAX_SIM_COSINE);
    index_struct_fp16.AddExtraParam(milvus::NLIST, "64");
    milvus::IndexDesc index_struct_bf16(combineStructName(field_struct, field_struct_bf16_vector), "index_bfloat16",
                                        milvus::IndexType::IVF_FLAT, milvus::MetricType::MAX_SIM_COSINE);
    index_struct_bf16.AddExtraParam(milvus::NLIST, "64");
    milvus::IndexDesc index_struct_int8(combineStructName(field_struct, field_struct_int8_vector), "index_int8",
                                        milvus::IndexType::HNSW, milvus::MetricType::MAX_SIM_L2);
    milvus::IndexDesc index_simplify(combineStructName(field_simplify_struct, field_struct_vector), "index_simplify",
                                     milvus::IndexType::HNSW, milvus::MetricType::L2);
    status = client->CreateIndex(milvus::CreateIndexRequest()
                                     .WithCollectionName(collection_name)
                                     .AddIndex(std::move(index_struct_float))
                                     .AddIndex(std::move(index_struct_binary))
                                     .AddIndex(std::move(index_struct_fp16))
                                     .AddIndex(std::move(index_struct_bf16))
                                     .AddIndex(std::move(index_struct_int8))
                                     .AddIndex(std::move(index_simplify)));
    util::CheckStatus("create index on struct vector fields", status);

    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection: " + std::string(collection_name), status);
}

void
insertData(milvus::MilvusClientV2Ptr& client) {
    const int64_t row_count = 100;
    milvus::EntityRows rows;
    for (auto i = 0; i < row_count; ++i) {
        milvus::EntityRow row;
        row[field_id] = i;
        row[field_name] = "film_" + std::to_string(i);

        std::vector<milvus::EntityRow> struct_list;
        for (auto k = 0; k < 5; ++k) {
            nlohmann::json st;
            st[field_struct_int32] = k;
            st[field_struct_varchar] = "clip_description_" + std::to_string(i);
            st[field_struct_vector] = util::GenerateFloatVector(float_dimension);
            st[field_struct_binary_vector] = util::GenerateBinaryVector(binary_dimension);
            st[field_struct_fp16_vector] = util::GenerateFloatVector(fp16_dimension);
            st[field_struct_bf16_vector] = util::GenerateFloatVector(fp16_dimension);
            st[field_struct_int8_vector] = util::GenerateInt8Vector(int8_dimension);
            struct_list.emplace_back(std::move(st));
        }
        row[field_struct] = struct_list;

        std::vector<milvus::EntityRow> simplify_struct_list;
        for (auto k = 0; k < 2; ++k) {
            nlohmann::json st;
            st[field_struct_vector] = util::GenerateFloatVector(simplify_dimension);
            simplify_struct_list.emplace_back(std::move(st));
        }
        row[field_simplify_struct] = simplify_struct_list;

        rows.emplace_back(std::move(row));
    }

    milvus::InsertResponse resp_insert;
    auto status = client->Insert(
        milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)), resp_insert);
    util::CheckStatus("insert", status);
    std::cout << resp_insert.Results().IdArray().IntIDArray().size() << " rows inserted by row-based." << std::endl;
}

milvus::EntityRows
queryStruct(milvus::MilvusClientV2Ptr& client, std::string filter) {
    auto request = milvus::QueryRequest()
                       .WithCollectionName(collection_name)
                       .WithFilter(filter)
                       .AddOutputField(field_struct)
                       .AddOutputField(field_simplify_struct)
                       .WithLimit(3)
                       .WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

    milvus::QueryResponse response;
    auto status = client->Query(request, response);
    util::CheckStatus("query", status);

    milvus::EntityRows output_rows;
    status = response.Results().OutputRows(output_rows);
    util::CheckStatus("get output rows", status);
    std::cout << "\nQuery results with filter: " << filter << std::endl;
    // for (const auto& row : output_rows) {
    //     std::cout << "\t" << row << std::endl;
    // }
    return output_rows;
}

// EmbeddingList search is to "find similar rows" by Maximum Similarity algorithm
void
embeddingListSearch(milvus::MilvusClientV2Ptr& client, std::string sub_field_name, std::string filter,
                    const std::function<void(milvus::EmbeddingList&, const nlohmann::json&)>& make_vector) {
    auto query_rows = queryStruct(client, std::move(filter));
    auto ann_field = combineStructName(field_struct, sub_field_name);

    auto request = milvus::SearchRequest()
                       .WithCollectionName(collection_name)
                       .WithLimit(3)
                       .WithAnnsField(ann_field)
                       .AddOutputField(field_name)
                       .AddOutputField(combineStructName(field_struct, field_struct_int32))
                       .AddOutputField(combineStructName(field_struct, field_struct_varchar));

    for (const auto& row : query_rows) {
        milvus::EmbeddingList emb_list;
        const auto& struct_list = row[field_struct];
        for (const auto& st : struct_list) {
            make_vector(emb_list, st);
        }
        request.AddEmbeddingList(std::move(emb_list));
    }

    milvus::SearchResponse response;
    auto status = client->Search(request, response);
    util::CheckStatus("search", status);

    std::cout << "\nEmbeddingList search on struct field's " << ann_field << ": " << ann_field << std::endl;
    for (auto& result : response.Results().Results()) {
        std::cout << "\nResult of one embedding list:" << std::endl;
        milvus::EntityRows output_rows;
        status = result.OutputRows(output_rows);
        for (const auto& result_row : output_rows) {
            std::cout << "\t" << result_row << std::endl;
        }
    }
}

// Element-level search is to "find similar structs in the struct field", each item of result corresponds to one struct
// in the struct field, and with a elementOffset value to indicate which struct it is.
void
elementLevelSearch(milvus::MilvusClientV2Ptr& client) {
    auto ann_field = combineStructName(field_simplify_struct, field_struct_vector);
    auto query_rows = queryStruct(client, "id == 5");

    // each row of SIMPLIFY_STRUCT_FIELD contains two structs, and each struct has a FLOAT_VECTOR_FIELD,
    // so we will have 2 vectors to search for each row
    std::vector<std::vector<float>> query_vectors;
    for (const auto& row : query_rows) {
        const auto& struct_list = row[field_simplify_struct];
        for (const auto& st : struct_list) {
            query_vectors.push_back(st[field_struct_vector].get<std::vector<float>>());
        }
    }

    auto request = milvus::SearchRequest()
                       .WithCollectionName(collection_name)
                       .WithLimit(3)
                       .WithAnnsField(ann_field)
                       .AddOutputField(field_name)
                       .WithFloatVectors(std::move(query_vectors));
    milvus::SearchResponse response;
    auto status = client->Search(request, response);
    util::CheckStatus("element-level search", status);

    // there will be two lists of search results corresponding to the two vectors in field_simplify_struct,
    // and each search result will have an additional field "element_offset" indicating which element
    // in the array field field_simplify_struct this result corresponds to
    std::cout << "\nElement-level search on " << ann_field << std::endl;
    for (auto& result : response.Results().Results()) {
        std::cout << "\nResult of one target vector:" << std::endl;
        milvus::EntityRows output_rows;
        status = result.OutputRows(output_rows);
        for (const auto& result_row : output_rows) {
            std::cout << "\t" << result_row << std::endl;
        }
    }
}

void
addCollectionStructField(milvus::MilvusClientV2Ptr& client) {
    std::cout << "\n===================================================" << std::endl;
    std::cout << "Add a new struct field to the existing collection" << std::endl;

    milvus::StructFieldSchema struct_schema =
        milvus::StructFieldSchema()
            .WithName(field_extra_struct)
            .WithDescription("additional metadata for films")
            .WithMaxCapacity(8)
            .WithNullable(true)
            .AddField(milvus::FieldSchema(field_extra_struct_rating, milvus::DataType::INT32))
            .AddField(milvus::FieldSchema(field_extra_struct_tag, milvus::DataType::VARCHAR).WithMaxLength(128));
    auto status = client->AddCollectionStructField(milvus::AddCollectionStructFieldRequest()
                                                       .WithCollectionName(collection_name)
                                                       .WithStructField(std::move(struct_schema)));
    util::CheckStatus("add collection struct field", status);
    std::cout << "Added struct field: " << field_extra_struct << std::endl;

    milvus::DescribeCollectionResponse desc_resp;
    status =
        client->DescribeCollection(milvus::DescribeCollectionRequest().WithCollectionName(collection_name), desc_resp);
    util::CheckStatus("describe collection", status);
    std::cout << desc_resp.Desc().Schema().StructFields().size() << " struct fields in collection after add"
              << std::endl;

    {
        auto request = milvus::QueryRequest()
                           .WithCollectionName(collection_name)
                           .WithFilter(std::string(field_id) + " == 5")
                           .AddOutputField(field_id)
                           .AddOutputField(field_extra_struct)
                           .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
        milvus::QueryResponse response;
        status = client->Query(request, response);
        util::CheckStatus("query existing row after addCollectionStructField", status);
        milvus::EntityRows output_rows;
        status = response.Results().OutputRows(output_rows);
        util::CheckStatus("get output rows", status);
        std::cout << "Existing row after addCollectionStructField() - new field is null" << std::endl;
        for (const auto& row : output_rows) {
            std::cout << "	" << row << std::endl;
        }
    }

    {
        milvus::EntityRow row;
        row[field_id] = 5000;
        row[field_name] = "film_5000";

        std::vector<milvus::EntityRow> struct_list;
        for (auto k = 0; k < 5; ++k) {
            nlohmann::json st;
            st[field_struct_int32] = k;
            st[field_struct_varchar] = "clip_description_5000";
            st[field_struct_vector] = util::GenerateFloatVector(float_dimension);
            st[field_struct_binary_vector] = util::GenerateBinaryVector(binary_dimension);
            st[field_struct_fp16_vector] = util::GenerateFloatVector(fp16_dimension);
            st[field_struct_bf16_vector] = util::GenerateFloatVector(fp16_dimension);
            st[field_struct_int8_vector] = util::GenerateInt8Vector(int8_dimension);
            struct_list.emplace_back(std::move(st));
        }
        row[field_struct] = struct_list;

        std::vector<milvus::EntityRow> simplify_struct_list;
        for (auto k = 0; k < 2; ++k) {
            nlohmann::json st;
            st[field_struct_vector] = util::GenerateFloatVector(simplify_dimension);
            simplify_struct_list.emplace_back(std::move(st));
        }
        row[field_simplify_struct] = simplify_struct_list;

        std::vector<milvus::EntityRow> extra_struct_list;
        {
            nlohmann::json st;
            st[field_extra_struct_rating] = 5;
            st[field_extra_struct_tag] = "favorite";
            extra_struct_list.emplace_back(std::move(st));
        }
        {
            nlohmann::json st;
            st[field_extra_struct_rating] = 4;
            st[field_extra_struct_tag] = "classic";
            extra_struct_list.emplace_back(std::move(st));
        }
        row[field_extra_struct] = extra_struct_list;

        milvus::InsertResponse insert_resp;
        status = client->Insert(milvus::InsertRequest().WithCollectionName(collection_name).AddRowData(std::move(row)),
                                insert_resp);
        util::CheckStatus("insert row with added struct field", status);
    }

    {
        auto request = milvus::QueryRequest()
                           .WithCollectionName(collection_name)
                           .WithFilter(std::string(field_id) + " == 5000")
                           .AddOutputField(field_id)
                           .AddOutputField(field_name)
                           .AddOutputField(field_extra_struct)
                           .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
        milvus::QueryResponse response;
        status = client->Query(request, response);
        util::CheckStatus("query new row with added struct field", status);
        milvus::EntityRows output_rows;
        status = response.Results().OutputRows(output_rows);
        util::CheckStatus("get output rows", status);
        std::cout << "New row with the added struct field" << std::endl;
        for (const auto& row : output_rows) {
            std::cout << "	" << row << std::endl;
        }
    }

    {
        auto request = milvus::QueryRequest()
                           .WithCollectionName(collection_name)
                           .WithFilter(std::string(field_id) + " == 5000")
                           .AddOutputField(field_id)
                           .AddOutputField(field_name)
                           .AddOutputField(combineStructName(field_extra_struct, field_extra_struct_rating))
                           .AddOutputField(combineStructName(field_extra_struct, field_extra_struct_tag))
                           .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
        milvus::QueryResponse response;
        status = client->Query(request, response);
        util::CheckStatus("query projected struct subfields", status);
        milvus::EntityRows output_rows;
        status = response.Results().OutputRows(output_rows);
        util::CheckStatus("get output rows", status);
        std::cout << "Projected subfields from the added struct field" << std::endl;
        for (const auto& row : output_rows) {
            std::cout << "	" << row << std::endl;
        }
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

    createCollection(client);
    insertData(client);

    {
        auto request = milvus::QueryRequest()
                           .WithCollectionName(collection_name)
                           .AddOutputField("count(*)")
                           .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        milvus::QueryResponse response;
        status = client->Query(request, response);
        util::CheckStatus("query count(*)", status);
        std::cout << "count(*) = " << response.Results().GetRowCount() << std::endl;
    }

    embeddingListSearch(client, field_struct_vector, "id in [0, 5]",
                        [](milvus::EmbeddingList& emb_list, const nlohmann::json& st) {
                            emb_list.AddFloatVector(st[field_struct_vector].get<std::vector<float>>());
                        });

    embeddingListSearch(client, field_struct_binary_vector, "id in [1, 6]",
                        [](milvus::EmbeddingList& emb_list, const nlohmann::json& st) {
                            emb_list.AddBinaryVector(st[field_struct_binary_vector].get<std::vector<uint8_t>>());
                        });

    embeddingListSearch(client, field_struct_fp16_vector, "id in [2, 7]",
                        [](milvus::EmbeddingList& emb_list, const nlohmann::json& st) {
                            emb_list.AddFloat16Vector(st[field_struct_fp16_vector].get<std::vector<float>>());
                        });

    embeddingListSearch(client, field_struct_bf16_vector, "id in [3, 8]",
                        [](milvus::EmbeddingList& emb_list, const nlohmann::json& st) {
                            emb_list.AddBFloat16Vector(st[field_struct_bf16_vector].get<std::vector<float>>());
                        });

    embeddingListSearch(client, field_struct_int8_vector, "id in [4, 9]",
                        [](milvus::EmbeddingList& emb_list, const nlohmann::json& st) {
                            emb_list.AddInt8Vector(st[field_struct_int8_vector].get<std::vector<int8_t>>());
                        });

    elementLevelSearch(client);
    addCollectionStructField(client);

    client->Disconnect();
}
