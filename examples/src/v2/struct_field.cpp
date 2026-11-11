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

// Demonstrates Milvus Array of Struct support on MilvusClientV2: multi-vector
// struct sub-fields, embedding-list and element-level search, element-level null
// (nullable sub-fields, nested Array sub-fields, whole null struct rows) and
// struct sub-field add/drop via bracket paths. The flow and data mirror the Java
// SDK StructExample.
//
// Requirements: a Milvus server with the element-null stack enabled (the SDK
// serializes element nullability, nested Array sub-fields and sub-field add/drop
// on the wire; an older server will reject these requests).

#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

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
const char* const field_struct_score = "score";
const char* const field_struct_lang = "lang";
const char* const field_struct_tags = "tags";
const char* const field_struct_note = "note";
const char* const field_struct_marks = "marks";
const char* const field_simplify_struct = "simplify_clips";
const char* const field_extra_struct = "metadata";
const char* const field_extra_struct_rating = "rating";
const char* const field_extra_struct_tag = "tag";
const int64_t extra_row_id = 5000;
const int64_t evolve_row_id = 6000;

// ids reserved for the element-null demo rows: they are inserted exactly once by
// buildElementNullRows() and skipped by the bulk insert below (no duplicate primary keys)
const std::set<int64_t> element_null_row_ids{97, 98, 99};

const uint32_t float_dimension = 128;
const uint32_t binary_dimension = 32;
const uint32_t fp16_dimension = 16;
const uint32_t bf16_dimension = 16;
const uint32_t int8_dimension = 32;
const uint32_t simplify_dimension = 32;
const int64_t struct_capacity = 100;

std::string
combineStructName(const std::string& struct_field, const std::string& sub_field) {
    return struct_field + "[" + sub_field + "]";
}

std::string
dataTypeName(milvus::DataType type) {
    switch (type) {
        case milvus::DataType::INT32:
            return "Int32";
        case milvus::DataType::INT64:
            return "Int64";
        case milvus::DataType::VARCHAR:
            return "VarChar";
        case milvus::DataType::ARRAY:
            return "Array";
        case milvus::DataType::FLOAT_VECTOR:
            return "FloatVector";
        case milvus::DataType::BINARY_VECTOR:
            return "BinaryVector";
        case milvus::DataType::FLOAT16_VECTOR:
            return "Float16Vector";
        case milvus::DataType::BFLOAT16_VECTOR:
            return "BFloat16Vector";
        case milvus::DataType::INT8_VECTOR:
            return "Int8Vector";
        default:
            return "Unknown";
    }
}

// Fill one clip struct element. score/lang are element-nullable (may be omitted or
// null) and tags is a nested Array whose leaves are element-nullable.
void
fillClip(nlohmann::json& st, int64_t id, int32_t frame, bool with_score, bool with_lang, const nlohmann::json& tags) {
    st[field_struct_int32] = frame;
    st[field_struct_varchar] = "clip_description_" + std::to_string(id);
    st[field_struct_vector] = util::GenerateFloatVector(float_dimension);
    st[field_struct_binary_vector] = util::GenerateBinaryVector(binary_dimension);
    st[field_struct_fp16_vector] = util::GenerateFloatVector(fp16_dimension);
    st[field_struct_bf16_vector] = util::GenerateFloatVector(bf16_dimension);
    st[field_struct_int8_vector] = util::GenerateInt8Vector(int8_dimension);
    if (with_score) {
        st[field_struct_score] = static_cast<int64_t>(id % 100 + frame);
    }
    if (with_lang) {
        st[field_struct_lang] = "en";
    }
    st[field_struct_tags] = tags;
}

void
createCollection(milvus::MilvusClientV2Ptr& client) {
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->SetEnableDynamicField(false);
    collection_schema->AddField(milvus::FieldSchema(field_id, milvus::DataType::INT64, "id", true, false));
    collection_schema->AddField(milvus::FieldSchema(field_name, milvus::DataType::VARCHAR).WithMaxLength(1024));

    // the clips struct field is nullable so whole null struct rows can be inserted/read back
    milvus::StructFieldSchema struct_schema =
        milvus::StructFieldSchema()
            .WithName(field_struct)
            .WithDescription("clips of a film")
            .WithMaxCapacity(struct_capacity)
            .WithNullable(true)
            .AddField(milvus::FieldSchema(field_struct_int32, milvus::DataType::INT32)
                          .WithDescription("from which frame this clip begin"))
            .AddField(milvus::FieldSchema(field_struct_varchar, milvus::DataType::VARCHAR)
                          .WithDescription("description of a clip")
                          .WithMaxLength(1024))
            .AddField(milvus::FieldSchema(field_struct_vector, milvus::DataType::FLOAT_VECTOR)
                          .WithDescription("float embedding of a clip")
                          .WithDimension(float_dimension))
            .AddField(milvus::FieldSchema(field_struct_binary_vector, milvus::DataType::BINARY_VECTOR)
                          .WithDescription("binary embedding of a clip")
                          .WithDimension(binary_dimension))
            .AddField(milvus::FieldSchema(field_struct_fp16_vector, milvus::DataType::FLOAT16_VECTOR)
                          .WithDimension(fp16_dimension))
            .AddField(milvus::FieldSchema(field_struct_bf16_vector, milvus::DataType::BFLOAT16_VECTOR)
                          .WithDimension(bf16_dimension))
            .AddField(milvus::FieldSchema(field_struct_int8_vector, milvus::DataType::INT8_VECTOR)
                          .WithDescription("int8 embedding of a clip")
                          .WithDimension(int8_dimension))
            .AddField(milvus::FieldSchema(field_struct_score, milvus::DataType::INT64)
                          .WithDescription("score of a clip, element-level null allowed")
                          .WithNullable(true))
            .AddField(milvus::FieldSchema(field_struct_lang, milvus::DataType::VARCHAR)
                          .WithDescription("language of a clip, element-level null allowed")
                          .WithMaxLength(16)
                          .WithNullable(true))
            .AddField(milvus::FieldSchema(field_struct_tags, milvus::DataType::ARRAY)
                          .WithDescription("tags of a clip, a nested array with null inner values allowed")
                          .WithElementType(milvus::DataType::VARCHAR)
                          .WithMaxCapacity(8)
                          .WithMaxLength(16)
                          .WithNullable(true)
                          .WithElementNullable(true));
    collection_schema->AddStructField(std::move(struct_schema));

    milvus::StructFieldSchema simplify_struct_schema =
        milvus::StructFieldSchema()
            .WithName(field_simplify_struct)
            .WithDescription("simplify clips")
            .WithMaxCapacity(struct_capacity)
            .AddField(milvus::FieldSchema(field_struct_vector, milvus::DataType::FLOAT_VECTOR)
                          .WithDescription("clip has been simplified")
                          .WithDimension(simplify_dimension));
    collection_schema->AddStructField(std::move(simplify_struct_schema));

    auto status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + std::string(collection_name), status);

    milvus::IndexParam index_struct_float(combineStructName(field_struct, field_struct_vector), "index_float",
                                          milvus::IndexType::HNSW, milvus::MetricType::MAX_SIM_IP);
    milvus::IndexParam index_struct_binary(combineStructName(field_struct, field_struct_binary_vector), "index_binary",
                                           milvus::IndexType::HNSW, milvus::MetricType::MAX_SIM_HAMMING);
    milvus::IndexParam index_struct_fp16(combineStructName(field_struct, field_struct_fp16_vector), "index_float16",
                                         milvus::IndexType::IVF_FLAT, milvus::MetricType::MAX_SIM_COSINE);
    index_struct_fp16.AddExtraParam(milvus::NLIST, "64");
    milvus::IndexParam index_struct_bf16(combineStructName(field_struct, field_struct_bf16_vector), "index_bfloat16",
                                         milvus::IndexType::IVF_FLAT, milvus::MetricType::MAX_SIM_COSINE);
    index_struct_bf16.AddExtraParam(milvus::NLIST, "64");
    milvus::IndexParam index_struct_int8(combineStructName(field_struct, field_struct_int8_vector), "index_int8",
                                         milvus::IndexType::HNSW, milvus::MetricType::MAX_SIM_L2);
    milvus::IndexParam index_simplify(combineStructName(field_simplify_struct, field_struct_vector), "index_simplify",
                                      milvus::IndexType::HNSW, milvus::MetricType::L2);
    status = client->CreateIndex(milvus::CreateIndexRequest()
                                     .WithCollectionName(collection_name)
                                     .AddIndexParam(std::move(index_struct_float))
                                     .AddIndexParam(std::move(index_struct_binary))
                                     .AddIndexParam(std::move(index_struct_fp16))
                                     .AddIndexParam(std::move(index_struct_bf16))
                                     .AddIndexParam(std::move(index_struct_int8))
                                     .AddIndexParam(std::move(index_simplify)));
    util::CheckStatus("create index on struct vector fields", status);

    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection: " + std::string(collection_name), status);

    std::cout << "Collection created: " << collection_name << std::endl;

    milvus::DescribeCollectionResponse desc_resp;
    status =
        client->DescribeCollection(milvus::DescribeCollectionRequest().WithCollectionName(collection_name), desc_resp);
    util::CheckStatus("describe collection", status);
    for (const auto& struct_field : desc_resp.Desc().Schema().StructFields()) {
        std::cout << "Struct field '" << struct_field.Name() << "' nullable=" << struct_field.IsNullable()
                  << " max_capacity=" << struct_field.MaxCapacity() << std::endl;
        for (const auto& sub_field : struct_field.Fields()) {
            std::cout << "  " << sub_field.Name() << ": " << dataTypeName(sub_field.FieldDataType())
                      << ", isNullable=" << sub_field.IsNullable()
                      << ", elementNullable=" << sub_field.IsElementNullable() << std::endl;
        }
    }
}

// Insert rowCount rows in batches of 300. Every 100th row stores a whole null struct
// row (clips = null); the ids in element_null_row_ids are skipped here and inserted
// exactly once by buildElementNullRows().
void
insertData(milvus::MilvusClientV2Ptr& client, int64_t row_count) {
    const int64_t batch_size = 300;
    milvus::EntityRows rows;
    for (int64_t id = 0; id < row_count; ++id) {
        if (element_null_row_ids.count(id) > 0) {
            continue;
        }
        milvus::EntityRow row;
        row[field_id] = id;
        row[field_name] = "film_" + std::to_string(id);

        if (id % 100 != 0) {
            std::vector<milvus::EntityRow> struct_list;
            for (auto k = 0; k < 5; ++k) {
                nlohmann::json st;
                fillClip(st, id, k, true, true, nlohmann::json::array({"a", "b"}));
                struct_list.emplace_back(std::move(st));
            }
            row[field_struct] = struct_list;
        } else {
            // the clips struct field is nullable; store a whole null struct row
            row[field_struct] = nullptr;
        }

        std::vector<milvus::EntityRow> simplify_struct_list;
        for (auto k = 0; k < 2; ++k) {
            nlohmann::json st;
            st[field_struct_vector] = util::GenerateFloatVector(simplify_dimension);
            simplify_struct_list.emplace_back(std::move(st));
        }
        row[field_simplify_struct] = simplify_struct_list;

        rows.emplace_back(std::move(row));
        if (static_cast<int64_t>(rows.size()) >= batch_size) {
            milvus::InsertResponse resp_insert;
            auto status = client->Insert(
                milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)), resp_insert);
            util::CheckStatus("insert", status);
            std::cout << "Inserted row count: " << resp_insert.Results().InsertCount() << std::endl;
            rows.clear();
        }
    }
    if (!rows.empty()) {
        milvus::InsertResponse resp_insert;
        auto status = client->Insert(
            milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)), resp_insert);
        util::CheckStatus("insert", status);
        std::cout << "Inserted row count: " << resp_insert.Results().InsertCount() << std::endl;
    }

    auto request = milvus::QueryRequest()
                       .WithCollectionName(collection_name)
                       .AddOutputField("count(*)")
                       .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    milvus::QueryResponse response;
    auto status = client->Query(request, response);
    util::CheckStatus("query count(*)", status);
    std::cout << response.Results().GetRowCount() << " rows persisted" << std::endl;
}

// Rows demonstrating element-level null: a missing or null sub-field key is stored as a
// null element for the element-nullable sub-fields (score/lang); the nested array supports
// null inner values (tags: ["c", null]) and empty arrays; the whole struct row can be null
// (clips: null).
std::vector<milvus::EntityRow>
buildElementNullRows() {
    std::vector<milvus::EntityRow> rows;
    {
        milvus::EntityRow row;
        row[field_id] = 97;
        row[field_name] = "film_97";
        std::vector<milvus::EntityRow> struct_list;
        {
            nlohmann::json st;
            fillClip(st, 97, 0, true, true, nlohmann::json::array({"a", "b"}));
            st[field_struct_score] = 1;
            struct_list.emplace_back(std::move(st));
        }
        {
            // element-level null for lang, and a null inner value in the nested array
            nlohmann::json st;
            fillClip(st, 97, 1, true, false, nlohmann::json::array({"c", nullptr}));
            st[field_struct_score] = 2;
            st[field_struct_lang] = nullptr;
            struct_list.emplace_back(std::move(st));
        }
        {
            // score/lang omitted => both stored as null elements; tags is an empty array
            nlohmann::json st;
            fillClip(st, 97, 2, false, false, nlohmann::json::array());
            struct_list.emplace_back(std::move(st));
        }
        row[field_struct] = struct_list;
        row[field_simplify_struct] = nlohmann::json::array();
        rows.emplace_back(std::move(row));
    }
    {
        milvus::EntityRow row;
        row[field_id] = 98;
        row[field_name] = "film_98";
        row[field_struct] = nlohmann::json::array();  // empty array, a non-null row
        row[field_simplify_struct] = nlohmann::json::array();
        rows.emplace_back(std::move(row));
    }
    {
        milvus::EntityRow row;
        row[field_id] = 99;
        row[field_name] = "film_99";
        row[field_struct] = nullptr;  // null struct row
        row[field_simplify_struct] = nlohmann::json::array();
        rows.emplace_back(std::move(row));
    }
    return rows;
}

void
verifyElementNull(milvus::MilvusClientV2Ptr& client) {
    std::cout << "\n===================================================" << std::endl;
    std::cout << "Verify element-level null struct fields on rows 97/98/99" << std::endl;

    auto request = milvus::QueryRequest()
                       .WithCollectionName(collection_name)
                       .WithFilter(std::string(field_id) + " in [97, 98, 99]")
                       .AddOutputField(field_id)
                       .AddOutputField(field_struct)
                       .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    milvus::QueryResponse response;
    auto status = client->Query(request, response);
    util::CheckStatus("query element-null rows", status);

    milvus::EntityRows rows;
    status = response.Results().OutputRows(rows);
    util::CheckStatus("get output rows", status);
    for (const auto& row : rows) {
        auto id = row[field_id].get<int64_t>();
        const auto& clips = row[field_struct];
        std::cout << "query result id=" << id << ", clips=" << clips.dump() << std::endl;
        if (id == 99 && !clips.is_null()) {
            throw std::runtime_error("Expected null clips for id=99, got: " + clips.dump());
        }
    }
    std::cout << "Element-level null verification passed" << std::endl;
}

void
verifyNullableStruct(milvus::MilvusClientV2Ptr& client) {
    std::cout << "\n===================================================" << std::endl;
    std::cout << "Verify nullable struct field '" << field_struct << "' returns null data" << std::endl;

    // rows with id % 100 == 0 were inserted with clips = null
    auto request = milvus::QueryRequest()
                       .WithCollectionName(collection_name)
                       .WithFilter(std::string(field_id) + " in [0, 100, 200, 300]")
                       .AddOutputField(field_id)
                       .AddOutputField(field_name)
                       .AddOutputField(field_struct)
                       .AddOutputField(field_simplify_struct)
                       .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    milvus::QueryResponse response;
    auto status = client->Query(request, response);
    util::CheckStatus("query nullable struct rows", status);

    milvus::EntityRows rows;
    status = response.Results().OutputRows(rows);
    util::CheckStatus("get output rows", status);
    for (const auto& row : rows) {
        auto id = row[field_id].get<int64_t>();
        const auto& clips = row[field_struct];
        const auto& simplify_clips = row[field_simplify_struct];
        std::cout << "query result id=" << id << ", clips=" << clips.dump()
                  << ", simplify_clips.size=" << simplify_clips.size() << std::endl;
        if (id % 100 == 0 && !clips.is_null()) {
            throw std::runtime_error("Expected null clips for id=" + std::to_string(id) + ", got: " + clips.dump());
        }
    }

    // take the simplify_clips vector of a null-clips row and search on it element-level;
    // the top-1 hit should be that row itself and its clips output must decode back to null
    auto vector_request = milvus::QueryRequest()
                              .WithCollectionName(collection_name)
                              .WithFilter(std::string(field_id) + " == 100")
                              .AddOutputField(field_simplify_struct)
                              .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    milvus::QueryResponse vector_response;
    status = client->Query(vector_request, vector_response);
    util::CheckStatus("query simplify_clips of a null-clips row", status);
    milvus::EntityRows vector_rows;
    status = vector_response.Results().OutputRows(vector_rows);
    util::CheckStatus("get output rows", status);
    const auto& simplify_structs = vector_rows.at(0)[field_simplify_struct];
    auto query_vector = simplify_structs.at(0)[field_struct_vector].get<std::vector<float>>();

    auto search_request = milvus::SearchRequest()
                              .WithCollectionName(collection_name)
                              .WithLimit(5)
                              .WithAnnsField(combineStructName(field_simplify_struct, field_struct_vector))
                              .WithMetricType(milvus::MetricType::L2)
                              .AddOutputField(field_id)
                              .AddOutputField(field_struct)
                              .WithFloatVectors({query_vector})
                              .WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);
    milvus::SearchResponse search_response;
    status = client->Search(search_request, search_response);
    util::CheckStatus("search back from a null-clips row", status);

    bool found_null_row = false;
    for (auto& result : search_response.Results().Results()) {
        milvus::EntityRows search_rows;
        status = result.OutputRows(search_rows);
        util::CheckStatus("get output rows", status);
        for (const auto& search_row : search_rows) {
            auto id = search_row[field_id].get<int64_t>();
            const auto& clips = search_row[field_struct];
            std::cout << "search result id=" << id << ", clips=" << clips.dump() << std::endl;
            if (id == 100) {
                if (!clips.is_null()) {
                    throw std::runtime_error("Expected null clips for id=100 in search result, got: " + clips.dump());
                }
                found_null_row = true;
            }
        }
    }
    if (!found_null_row) {
        throw std::runtime_error("The null-clips row id=100 was not returned by search");
    }
    std::cout << "Nullable struct verification passed" << std::endl;
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
    std::cout << "\nQuery with filter expression: " << filter << std::endl;
    for (const auto& row : output_rows) {
        std::cout << "\t" << row << std::endl;
    }
    return output_rows;
}

// Embedding-list search is to "find similar rows" by Maximum Similarity algorithm. Each
// query row contributes one EmbeddingList built from its clips elements; a row whose
// clips field is a null struct row has no embeddings and is skipped.
void
search(milvus::MilvusClientV2Ptr& client, std::string sub_field_name, std::string filter,
       const std::function<void(milvus::EmbeddingList&, const nlohmann::json&)>& make_vector) {
    auto query_rows = queryStruct(client, std::move(filter));
    auto ann_field = combineStructName(field_struct, sub_field_name);

    auto request = milvus::SearchRequest()
                       .WithCollectionName(collection_name)
                       .WithLimit(5)
                       .WithAnnsField(ann_field)
                       .AddOutputField(field_name)
                       .AddOutputField(combineStructName(field_struct, field_struct_int32))
                       .AddOutputField(combineStructName(field_struct, field_struct_varchar))
                       .AddOutputField(combineStructName(field_simplify_struct, field_struct_vector));

    int64_t nq = 0;
    for (const auto& row : query_rows) {
        const auto& struct_list = row[field_struct];
        if (struct_list.is_null()) {
            // this row holds a null (nullable) struct field, so it has no embeddings to search on
            continue;
        }
        milvus::EmbeddingList emb_list;
        for (const auto& st : struct_list) {
            make_vector(emb_list, st);
        }
        request.AddEmbeddingList(std::move(emb_list));
        ++nq;
    }
    if (nq == 0) {
        std::cout << "No embedding lists to search for field '" << sub_field_name << "'" << std::endl;
        return;
    }

    std::cout << "\nSearch on field '" << sub_field_name << "' in struct '" << field_struct << "' with nq=" << nq
              << std::endl;
    milvus::SearchResponse response;
    auto status = client->Search(request, response);
    util::CheckStatus("search", status);

    int i = 0;
    for (auto& result : response.Results().Results()) {
        std::cout << "Results of No." << i++ << " embedding list" << std::endl;
        milvus::EntityRows output_rows;
        status = result.OutputRows(output_rows);
        for (const auto& result_row : output_rows) {
            std::cout << "\t" << result_row << std::endl;
        }
    }
}

void
searchFloatVectorField(milvus::MilvusClientV2Ptr& client) {
    search(client, field_struct_vector, "id in [5, 8]", [](milvus::EmbeddingList& emb_list, const nlohmann::json& st) {
        emb_list.AddFloatVector(st[field_struct_vector].get<std::vector<float>>());
    });
}

void
searchBinaryVectorField(milvus::MilvusClientV2Ptr& client) {
    auto filter = std::string("MATCH_LEAST(") + field_struct + ", $[" + field_struct_int32 + "] < 2000, threshold=3)";
    search(client, field_struct_binary_vector, filter, [](milvus::EmbeddingList& emb_list, const nlohmann::json& st) {
        emb_list.AddBinaryVector(st[field_struct_binary_vector].get<std::vector<uint8_t>>());
    });
}

void
searchFp16VectorField(milvus::MilvusClientV2Ptr& client) {
    search(client, field_struct_fp16_vector, "id in [3, 9]",
           [](milvus::EmbeddingList& emb_list, const nlohmann::json& st) {
               emb_list.AddFloat16Vector(st[field_struct_fp16_vector].get<std::vector<float>>());
           });
}

void
searchBf16VectorField(milvus::MilvusClientV2Ptr& client) {
    search(client, field_struct_bf16_vector, "id in [4, 9]",
           [](milvus::EmbeddingList& emb_list, const nlohmann::json& st) {
               emb_list.AddBFloat16Vector(st[field_struct_bf16_vector].get<std::vector<float>>());
           });
}

void
searchInt8VectorField(milvus::MilvusClientV2Ptr& client) {
    search(client, field_struct_int8_vector, "id == 999",
           [](milvus::EmbeddingList& emb_list, const nlohmann::json& st) {
               emb_list.AddInt8Vector(st[field_struct_int8_vector].get<std::vector<int8_t>>());
           });
}

// Element-level search is to "find similar structs in the struct field": each item of the
// result corresponds to one struct in the struct field, and carries an element offset.
void
elementLevelSearch(milvus::MilvusClientV2Ptr& client) {
    std::cout << "\n===================================================" << std::endl;
    auto ann_field = combineStructName(field_simplify_struct, field_struct_vector);
    std::cout << "Element-level search on field '" << field_struct_vector << "' in struct '" << field_simplify_struct
              << "'" << std::endl;

    auto query_request = milvus::QueryRequest()
                             .WithCollectionName(collection_name)
                             .WithFilter(std::string(field_id) + " == 5")
                             .AddOutputField(field_simplify_struct)
                             .WithLimit(1)
                             .WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);
    milvus::QueryResponse query_response;
    auto status = client->Query(query_request, query_response);
    util::CheckStatus("query simplify_clips", status);
    milvus::EntityRows query_rows;
    status = query_response.Results().OutputRows(query_rows);
    util::CheckStatus("get output rows", status);
    const auto& struct_list = query_rows.at(0)[field_simplify_struct];

    // each row of SIMPLIFY_STRUCT_FIELD contains two structs, and each struct has a
    // FLOAT_VECTOR_FIELD, so we will have 2 vectors to search for each row
    std::vector<std::vector<float>> query_vectors;
    for (const auto& st : struct_list) {
        query_vectors.push_back(st[field_struct_vector].get<std::vector<float>>());
    }

    auto request = milvus::SearchRequest()
                       .WithCollectionName(collection_name)
                       .WithLimit(3)
                       .WithAnnsField(ann_field)
                       .WithMetricType(milvus::MetricType::L2)
                       .AddOutputField(field_name)
                       .WithFloatVectors(std::move(query_vectors))
                       .WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);
    milvus::SearchResponse response;
    status = client->Search(request, response);
    util::CheckStatus("element-level search", status);

    // there will be two lists of search results corresponding to the two vectors in the
    // simplify_clips field, and each search result carries an element offset indicating which
    // element in the array field this result corresponds to
    int i = 0;
    for (auto& result : response.Results().Results()) {
        std::cout << "Results of No." << i++ << " query element" << std::endl;
        milvus::EntityRows output_rows;
        status = result.OutputRows(output_rows);
        for (const auto& result_row : output_rows) {
            std::cout << "\t" << result_row << std::endl;
        }
    }
}

// Element-level query: the iterator expands each matched element into one row carrying an
// "offset" column (the matched element's index within the struct-array field).
void
elementLevelQuery(milvus::MilvusClientV2Ptr& client) {
    std::cout << "\n===================================================" << std::endl;
    auto filter = std::string("element_filter(") + field_struct + ", $[" + field_struct_int32 + "] < 2000)";
    std::cout << "Element-level query with filter expression: " << filter << std::endl;

    milvus::QueryIteratorRequest request;
    request.SetCollectionName(collection_name);
    request.SetBatchSize(100);
    request.SetFilter(filter);
    request.AddOutputField(field_id);
    request.AddOutputField(field_name);
    request.AddOutputField(combineStructName(field_struct, field_struct_int32));
    request.AddOutputField(combineStructName(field_struct, field_struct_varchar));

    milvus::QueryIteratorPtr iterator;
    auto status = client->QueryIterator(request, iterator);
    util::CheckStatus("get element-filter query iterator", status);

    uint64_t total_elements = 0;
    while (true) {
        milvus::QueryResults batch_results;
        status = iterator->Next(batch_results);
        util::CheckStatus("iterator next batch", status);
        auto batch_count = batch_results.GetRowCount();
        if (batch_count == 0) {
            break;
        }
        total_elements += batch_count;
        milvus::EntityRows rows;
        status = batch_results.OutputRows(rows);
        util::CheckStatus("get output rows", status);
        for (const auto& row : rows) {
            auto id = row[field_id].get<int64_t>();
            auto frame = row[combineStructName(field_struct, field_struct_int32)].get<int32_t>();
            auto offset = row.value("offset", int64_t{0});
            std::cout << "  id=" << id << ", frame_number=" << frame << ", elementOffset=" << offset << std::endl;
        }
    }
    std::cout << "Total matched elements: " << total_elements << std::endl;
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
        row[field_id] = extra_row_id;
        row[field_name] = "film_5000";

        std::vector<milvus::EntityRow> struct_list;
        for (auto k = 0; k < 5; ++k) {
            nlohmann::json st;
            fillClip(st, extra_row_id, k, true, true, nlohmann::json::array({"a", "b"}));
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
                           .WithFilter(std::string(field_id) + " == " + std::to_string(extra_row_id))
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
                           .WithFilter(std::string(field_id) + " == " + std::to_string(extra_row_id))
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

// Evolve an existing struct field: add sub-fields (including a nested Array sub-field),
// insert rows carrying them, and drop one again. Sub-field add/drop use the bracketed
// struct[sub] path.
void
evolveStructSubField(milvus::MilvusClientV2Ptr& client) {
    std::cout << "\n===================================================" << std::endl;
    std::cout << "Evolve struct field '" << field_struct << "' by adding sub-fields" << std::endl;

    // sub-fields must be element-nullable
    milvus::FieldSchema note(combineStructName(field_struct, field_struct_note), milvus::DataType::VARCHAR);
    note.WithMaxLength(64).WithNullable(true);
    auto status = client->AddCollectionField(
        milvus::AddCollectionFieldRequest().WithCollectionName(collection_name).WithField(std::move(note)));
    util::CheckStatus("add struct sub-field clips[note]", status);
    std::cout << "Added struct sub-field " << field_struct << "[" << field_struct_note << "]" << std::endl;

    // a nested Array sub-field add requires max_capacity and describes null leaves
    milvus::FieldSchema marks(combineStructName(field_struct, field_struct_marks), milvus::DataType::ARRAY);
    marks.WithElementType(milvus::DataType::INT32).WithMaxCapacity(4).WithNullable(true).WithElementNullable(true);
    status = client->AddCollectionField(
        milvus::AddCollectionFieldRequest().WithCollectionName(collection_name).WithField(std::move(marks)));
    util::CheckStatus("add nested Array struct sub-field clips[marks]", status);
    std::cout << "Added nested Array struct sub-field " << field_struct << "[" << field_struct_marks << "]"
              << std::endl;

    {
        milvus::EntityRow row;
        row[field_id] = evolve_row_id;
        row[field_name] = "film_6000";
        std::vector<milvus::EntityRow> struct_list;
        {
            nlohmann::json st;
            fillClip(st, evolve_row_id, 0, true, true, nlohmann::json::array({"a", "b"}));
            st[field_struct_score] = 7;
            st[field_struct_note] = "hello";
            st[field_struct_marks] = nlohmann::json::array({1, 2});
            struct_list.emplace_back(std::move(st));
        }
        row[field_struct] = struct_list;
        row[field_simplify_struct] = nlohmann::json::array();
        milvus::InsertResponse resp;
        status = client->Insert(milvus::InsertRequest().WithCollectionName(collection_name).AddRowData(std::move(row)),
                                resp);
        util::CheckStatus("insert after add struct sub-fields", status);
    }

    milvus::DescribeCollectionResponse desc_resp;
    status =
        client->DescribeCollection(milvus::DescribeCollectionRequest().WithCollectionName(collection_name), desc_resp);
    util::CheckStatus("describe collection", status);
    std::cout << "Collection fields after sub-field evolution:" << std::endl;
    for (const auto& struct_field : desc_resp.Desc().Schema().StructFields()) {
        if (struct_field.Name() != field_struct) {
            continue;
        }
        for (const auto& sub_field : struct_field.Fields()) {
            std::cout << "  " << sub_field.Name() << ": " << dataTypeName(sub_field.FieldDataType())
                      << ", isNullable=" << sub_field.IsNullable()
                      << ", elementNullable=" << sub_field.IsElementNullable() << std::endl;
        }
    }

    status = client->DropCollectionField(milvus::DropCollectionFieldRequest()
                                             .WithCollectionName(collection_name)
                                             .WithFieldName(combineStructName(field_struct, field_struct_note)));
    util::CheckStatus("drop struct sub-field clips[note]", status);
    std::cout << "Dropped struct sub-field " << field_struct << "[" << field_struct_note << "]" << std::endl;
}

}  // namespace

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClientV2::Create();

    milvus::ConnectParam connect_param{"http://localhost:19530", "root:Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    try {
        createCollection(client);
        insertData(client, 2000);

        // insert rows demonstrating element-level null and null struct rows. ids
        // 97/98/99 were skipped by the bulk insert, so each is inserted only once here
        auto element_null_rows = buildElementNullRows();
        milvus::InsertResponse element_resp;
        status = client->Insert(
            milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(element_null_rows)),
            element_resp);
        util::CheckStatus("insert element-null rows", status);
        verifyElementNull(client);

        verifyNullableStruct(client);

        searchFloatVectorField(client);
        searchBinaryVectorField(client);
        searchFp16VectorField(client);
        searchBf16VectorField(client);
        searchInt8VectorField(client);
        elementLevelSearch(client);
        elementLevelQuery(client);

        addCollectionStructField(client);
        evolveStructSubField(client);

        status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
        util::CheckStatus("drop collection: " + std::string(collection_name), status);
        std::cout << "Collection dropped: " << collection_name << std::endl;
    } catch (const std::exception& e) {
        // a failed step must not be masked as a success for run_examples.sh; drop the demo
        // collection before exiting with a non-zero code
        std::cout << "Example failed: " << e.what() << std::endl;
        client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
        client->Disconnect();
        return 1;
    }

    client->Disconnect();
    return 0;
}
