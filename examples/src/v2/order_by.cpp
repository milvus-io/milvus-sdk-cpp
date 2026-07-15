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

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {

const char* const collection_name = "cpp_sdk_example_order_by_v2";
const char* const id_field = "id";
const char* const price_field = "price";
const char* const rating_field = "rating";
const char* const category_field = "category";
const char* const metadata_field = "metadata";
const char* const vector_field = "embeddings";
const char* const dynamic_views_field = "dynamic_views";

constexpr uint32_t dimension = 8;
constexpr int64_t entity_count = 200;

std::mt19937 random_generator(19530);
std::uniform_real_distribution<float> vector_distribution(0.0f, 1.0f);
std::uniform_real_distribution<double> rating_distribution(0.0, 5.0);
std::uniform_int_distribution<int64_t> views_distribution(0, 99);

std::vector<float>
randomVector() {
    std::vector<float> vector(dimension);
    for (auto& value : vector) {
        value = vector_distribution(random_generator);
    }
    return vector;
}

void
printTitle(const std::string& title) {
    std::cout << "\n============================================================\n"
              << title << "\n============================================================" << std::endl;
}

void
printSearchResults(const milvus::SearchResponse& response, const std::string& title) {
    printTitle(title);
    const auto& results = response.Results().Results();
    for (size_t query_index = 0; query_index < results.size(); ++query_index) {
        std::cout << "Search result " << query_index << ":" << std::endl;
        milvus::EntityRows rows;
        auto status = results.at(query_index).OutputRows(rows);
        util::CheckStatus("read search output rows", status);
        const auto& scores = results.at(query_index).Scores();
        for (size_t row_index = 0; row_index < rows.size(); ++row_index) {
            auto output_fields = rows.at(row_index);
            output_fields.erase("score");
            std::cout << "  {id: " << output_fields.at(id_field) << ", Score: " << std::fixed << std::setprecision(4)
                      << scores.at(row_index) << ", OutputFields: " << output_fields << "}" << std::endl;
        }
    }
}

void
printQueryResults(const milvus::QueryResponse& response, const std::string& title) {
    printTitle(title);
    milvus::EntityRows rows;
    auto status = response.Results().OutputRows(rows);
    util::CheckStatus("read query output rows", status);
    for (const auto& row : rows) {
        std::cout << "  " << row << std::endl;
    }
}

void
prepareCollection(milvus::MilvusClientV2Ptr& client) {
    client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));

    auto schema = std::make_shared<milvus::CollectionSchema>();
    schema->SetEnableDynamicField(true);
    schema->AddField({id_field, milvus::DataType::INT64, "", true, false});
    schema->AddField({price_field, milvus::DataType::DOUBLE});
    schema->AddField({rating_field, milvus::DataType::DOUBLE});
    schema->AddField(milvus::FieldSchema(category_field, milvus::DataType::VARCHAR).WithMaxLength(64));
    schema->AddField({metadata_field, milvus::DataType::JSON});
    schema->AddField(milvus::FieldSchema(vector_field, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));

    milvus::IndexDesc index(vector_field, "", milvus::IndexType::IVF_FLAT, milvus::MetricType::L2);
    index.AddExtraParam("nlist", "128");
    auto request = milvus::CreateCollectionRequest()
                       .WithCollectionName(collection_name)
                       .WithCollectionSchema(schema)
                       .WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED)
                       .AddIndex(std::move(index));
    auto status = client->CreateCollection(request);
    util::CheckStatus(std::string("create collection: ") + collection_name, status);

    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name).WithSync(true));
    util::CheckStatus(std::string("load collection: ") + collection_name, status);
    std::cout << "Collection created" << std::endl;
}

void
insertRows(milvus::MilvusClientV2Ptr& client) {
    const std::vector<std::string> categories = {"cate1", "cate2", "cate3", "cate4", "cate5"};
    milvus::EntityRows rows;
    rows.reserve(entity_count);
    for (int64_t i = 0; i < entity_count; ++i) {
        milvus::EntityRow row;
        row[id_field] = i;
        row[price_field] = 10.0 + static_cast<double>(i % 12);
        row[rating_field] = std::round(rating_distribution(random_generator) * 10.0) / 10.0;
        row[category_field] = categories.at(i % categories.size());
        row[metadata_field] = {{"age", 18 + (i % 40)},
                               {"score", i % 101},
                               {"popularity", std::round(rating_distribution(random_generator) * 20.0) / 10.0},
                               {"tags", {categories.at(i % categories.size()), "tag_" + std::to_string(i % 10)}}};
        row[vector_field] = randomVector();
        row[dynamic_views_field] = i * 10 + views_distribution(random_generator);
        rows.emplace_back(std::move(row));
    }

    milvus::InsertResponse response;
    auto status = client->Insert(
        milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)), response);
    util::CheckStatus("insert rows", status);
    std::cout << entity_count << " rows inserted" << std::endl;

    auto count_request = milvus::QueryRequest()
                             .WithCollectionName(collection_name)
                             .AddOutputField("count(*)")
                             .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    milvus::QueryResponse count_response;
    status = client->Query(count_request, count_response);
    util::CheckStatus("query count(*)", status);
    milvus::EntityRows count_rows;
    status = count_response.Results().OutputRows(count_rows);
    util::CheckStatus("read count(*)", status);
    std::cout << count_rows.at(0).at("count(*)") << " rows persisted" << std::endl;
}

void
searchExamples(milvus::MilvusClientV2Ptr& client) {
    const auto query_vector = randomVector();
    auto search_by_price = milvus::SearchRequest()
                               .WithCollectionName(collection_name)
                               .WithAnnsField(vector_field)
                               .AddFloatVector(query_vector)
                               .WithMetricType(milvus::MetricType::L2)
                               .WithLimit(10)
                               .AddOutputField(id_field)
                               .AddOutputField(price_field)
                               .AddOutputField(rating_field)
                               .AddOutputField(category_field)
                               .AddOrderByField(milvus::OrderByField(price_field, milvus::AggregationDirection::ASC))
                               .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    milvus::SearchResponse search_by_price_response;
    auto status = client->Search(search_by_price, search_by_price_response);
    util::CheckStatus("search ordered by price", status);
    printSearchResults(search_by_price_response, "Search with order_by_fields price ASC");

    auto search_by_price_and_rating =
        milvus::SearchRequest()
            .WithCollectionName(collection_name)
            .WithAnnsField(vector_field)
            .AddFloatVector(query_vector)
            .WithMetricType(milvus::MetricType::L2)
            .WithLimit(10)
            .AddOutputField(id_field)
            .AddOutputField(price_field)
            .AddOutputField(rating_field)
            .AddOutputField(category_field)
            .AddOutputField(metadata_field)
            .WithOrderByFields({milvus::OrderByField(price_field, milvus::AggregationDirection::ASC),
                                milvus::OrderByField(rating_field, milvus::AggregationDirection::DESC)})
            .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    milvus::SearchResponse search_by_price_and_rating_response;
    status = client->Search(search_by_price_and_rating, search_by_price_and_rating_response);
    util::CheckStatus("search ordered by price and rating", status);
    printSearchResults(search_by_price_and_rating_response, "Search with order_by_fields price ASC, rating DESC");

    auto search_by_json_and_dynamic =
        milvus::SearchRequest()
            .WithCollectionName(collection_name)
            .WithAnnsField(vector_field)
            .AddFloatVector(query_vector)
            .WithMetricType(milvus::MetricType::L2)
            .WithLimit(10)
            .AddOutputField(id_field)
            .AddOutputField(price_field)
            .AddOutputField(rating_field)
            .AddOutputField(category_field)
            .AddOutputField(metadata_field)
            .AddOutputField(dynamic_views_field)
            .WithOrderByFields({milvus::OrderByField(R"(metadata["age"])", milvus::AggregationDirection::ASC),
                                milvus::OrderByField(dynamic_views_field, milvus::AggregationDirection::DESC)})
            .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);
    milvus::SearchResponse search_by_json_and_dynamic_response;
    status = client->Search(search_by_json_and_dynamic, search_by_json_and_dynamic_response);
    util::CheckStatus("search ordered by JSON path and dynamic field", status);
    printSearchResults(search_by_json_and_dynamic_response,
                       R"(Search with order_by_fields metadata["age"] ASC, dynamic_views DESC)");
}

void
queryExamples(milvus::MilvusClientV2Ptr& client) {
    auto query_by_price = milvus::QueryRequest()
                              .WithCollectionName(collection_name)
                              .WithFilter("id >= 0")
                              .WithLimit(30)
                              .AddOutputField(id_field)
                              .AddOutputField(price_field)
                              .AddOutputField(rating_field)
                              .AddOutputField(category_field)
                              .AddOrderByField(milvus::OrderByField(price_field, milvus::AggregationDirection::DESC));
    milvus::QueryResponse query_by_price_response;
    auto status = client->Query(query_by_price, query_by_price_response);
    util::CheckStatus("query ordered by price", status);
    printQueryResults(query_by_price_response, "Query with order_by_fields price DESC");

    auto query_by_price_and_rating =
        milvus::QueryRequest()
            .WithCollectionName(collection_name)
            .WithFilter("id >= 0")
            .WithLimit(30)
            .AddOutputField(id_field)
            .AddOutputField(price_field)
            .AddOutputField(rating_field)
            .AddOutputField(category_field)
            .AddOutputField(metadata_field)
            .WithOrderByFields({milvus::OrderByField(price_field, milvus::AggregationDirection::ASC),
                                milvus::OrderByField(rating_field, milvus::AggregationDirection::DESC)});
    milvus::QueryResponse query_by_price_and_rating_response;
    status = client->Query(query_by_price_and_rating, query_by_price_and_rating_response);
    util::CheckStatus("query ordered by price and rating", status);
    printQueryResults(query_by_price_and_rating_response, "Query with order_by_fields price ASC, rating DESC");

    auto query_by_category_and_price =
        milvus::QueryRequest()
            .WithCollectionName(collection_name)
            .WithFilter("id >= 0")
            .WithLimit(30)
            .AddOutputField(id_field)
            .AddOutputField(price_field)
            .AddOutputField(rating_field)
            .AddOutputField(category_field)
            .AddOutputField(metadata_field)
            .AddOutputField(dynamic_views_field)
            .WithOrderByFields({milvus::OrderByField(category_field, milvus::AggregationDirection::ASC),
                                milvus::OrderByField(price_field, milvus::AggregationDirection::DESC)});
    milvus::QueryResponse query_by_category_and_price_response;
    status = client->Query(query_by_category_and_price, query_by_category_and_price_response);
    util::CheckStatus("query ordered by category and price", status);
    printQueryResults(query_by_category_and_price_response, "Query with order_by_fields category ASC, price DESC");
}

}  // namespace

int
main() {
    auto client = milvus::MilvusClientV2::Create();
    auto status = client->Connect({"http://localhost:19530", "root:Milvus"});
    util::CheckStatus("connect milvus server", status);

    prepareCollection(client);
    insertRows(client);
    searchExamples(client);
    queryExamples(client);

    client->Disconnect();
    std::cout << "All order_by examples completed!" << std::endl;
    return 0;
}
