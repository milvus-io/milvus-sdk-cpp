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

#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {

const char* const collection_name = "CPP_V2_SEARCH_AGGREGATION";
const char* const field_id = "id";
const char* const field_category = "category";
const char* const field_brand = "brand";
const char* const field_color = "color";
const char* const field_sku = "sku";
const char* const field_price = "price";
const char* const field_rating = "rating";
const char* const field_in_stock = "in_stock";
const char* const field_meta = "meta";
const char* const field_embedding = "embedding";

constexpr uint32_t dimension = 8;
constexpr int64_t entity_count = 2000;
constexpr int query_count = 3;
constexpr int64_t search_limit = 10;
constexpr size_t indent_step = 4;

const std::vector<std::string> categories = {"electronics", "books", "clothing", "home", "toys"};
const std::vector<std::string> brands = {"BrandA", "BrandB", "BrandC", "BrandD",
                                         "BrandE", "BrandF", "BrandG", "BrandH"};
const std::vector<std::string> colors = {"red", "blue", "green", "black"};
const std::vector<std::string> skus = {
    "sku_000", "sku_001", "sku_002", "sku_003", "sku_004", "sku_005", "sku_006", "sku_007", "sku_008", "sku_009",
    "sku_010", "sku_011", "sku_012", "sku_013", "sku_014", "sku_015", "sku_016", "sku_017", "sku_018", "sku_019",
};

std::mt19937 random_generator(19530);
std::uniform_real_distribution<float> random_float(0.0f, 1.0f);

std::vector<float>
randomVector() {
    std::vector<float> vector(dimension);
    for (auto& value : vector) {
        value = random_float(random_generator);
    }
    return vector;
}

void
addQueryVectors(milvus::SearchRequest& request) {
    for (int i = 0; i < query_count; ++i) {
        request.AddFloatVector(randomVector());
    }
}

std::string
formatValue(const nlohmann::json& value) {
    return value.is_string() ? value.get<std::string>() : value.dump();
}

std::string
formatKey(const std::vector<milvus::AggregationBucketKey>& key_entries) {
    std::ostringstream stream;
    for (size_t i = 0; i < key_entries.size(); ++i) {
        if (i > 0) {
            stream << ", ";
        }
        const auto& entry = key_entries[i];
        const auto name = entry.field_name.empty() ? std::to_string(entry.field_id) : entry.field_name;
        stream << name << "=" << formatValue(entry.value);
    }
    return stream.str();
}

nlohmann::json
toJson(const std::unordered_map<std::string, nlohmann::json>& values) {
    nlohmann::json object = nlohmann::json::object();
    for (const auto& item : values) {
        object[item.first] = item.second;
    }
    return object;
}

void
printBucketsRecursive(const std::vector<milvus::AggregationBucket>& buckets, size_t depth) {
    const std::string bucket_pad(depth * indent_step, ' ');
    const std::string label_pad(depth * indent_step + indent_step / 2, ' ');
    const std::string hit_pad(depth * indent_step + indent_step, ' ');
    const char* marker = depth == 0 ? "*" : "\\-";

    for (const auto& bucket : buckets) {
        std::cout << bucket_pad << marker << " [L" << depth << "] key[" << formatKey(bucket.key)
                  << "] count=" << bucket.count;
        if (!bucket.metrics.empty()) {
            std::cout << "  metrics=" << toJson(bucket.metrics).dump();
        }
        std::cout << std::endl;

        if (!bucket.hits.empty()) {
            std::cout << label_pad << "top_hits:" << std::endl;
            for (const auto& hit : bucket.hits) {
                std::cout << hit_pad << "- pk=" << formatValue(hit.id) << " score=" << std::fixed
                          << std::setprecision(4) << hit.score << "  fields=" << toJson(hit.fields).dump() << std::endl;
            }
        }
        if (!bucket.sub_groups.empty()) {
            std::cout << label_pad << "sub_groups:" << std::endl;
            printBucketsRecursive(bucket.sub_groups, depth + 1);
        }
    }
}

void
printBuckets(const std::string& label, const milvus::AggregationBuckets& per_query_buckets) {
    std::cout << "\n=== " << label << " ===" << std::endl;
    for (size_t i = 0; i < per_query_buckets.size(); ++i) {
        const auto& buckets = per_query_buckets[i];
        std::cout << "--- nq[" << i << "] (" << buckets.size() << " buckets) ---" << std::endl;
        printBucketsRecursive(buckets, 0);
    }
}

void
executeSearch(milvus::MilvusClientV2Ptr& client, const std::string& label, const milvus::SearchRequest& request) {
    milvus::SearchResponse response;
    auto status = client->Search(request, response);
    util::CheckStatus("search: " + label, status);
    printBuckets(label, response.AggregationBuckets());
}

void
buildCollection(milvus::MilvusClientV2Ptr& client) {
    client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));

    auto schema = std::make_shared<milvus::CollectionSchema>();
    schema->AddField({field_id, milvus::DataType::INT64, "", true, false});
    schema->AddField(milvus::FieldSchema(field_category, milvus::DataType::VARCHAR).WithMaxLength(32));
    schema->AddField(milvus::FieldSchema(field_brand, milvus::DataType::VARCHAR).WithMaxLength(32));
    schema->AddField(milvus::FieldSchema(field_color, milvus::DataType::VARCHAR).WithMaxLength(16));
    schema->AddField(milvus::FieldSchema(field_sku, milvus::DataType::VARCHAR).WithMaxLength(16));
    schema->AddField({field_price, milvus::DataType::DOUBLE});
    schema->AddField({field_rating, milvus::DataType::DOUBLE});
    schema->AddField({field_in_stock, milvus::DataType::BOOL});
    schema->AddField({field_meta, milvus::DataType::JSON});
    schema->AddField(milvus::FieldSchema(field_embedding, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));

    milvus::IndexDesc vector_index(field_embedding, "", milvus::IndexType::IVF_FLAT, milvus::MetricType::L2);
    vector_index.AddExtraParam("nlist", "128");
    auto create_request = milvus::CreateCollectionRequest()
                              .WithCollectionName(collection_name)
                              .WithCollectionSchema(schema)
                              .WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED)
                              .AddIndex(std::move(vector_index));
    auto status = client->CreateCollection(create_request);
    util::CheckStatus("create collection: " + std::string(collection_name), status);

    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name).WithSync(true));
    util::CheckStatus("load collection: " + std::string(collection_name), status);

    milvus::EntityRows rows;
    rows.reserve(entity_count);
    for (int64_t i = 0; i < entity_count; ++i) {
        milvus::EntityRow row;
        row[field_id] = i;
        row[field_category] = categories[i % categories.size()];
        row[field_brand] = brands[i % brands.size()];
        row[field_color] = colors[(i / brands.size()) % colors.size()];
        row[field_sku] = skus[(i / (brands.size() * colors.size())) % skus.size()];
        row[field_price] = 10.0 + static_cast<double>(i % 500);
        row[field_rating] = static_cast<double>(i % 50) / 10.0;
        row[field_in_stock] = i % 3 != 0;
        row[field_meta] = {{"subcategory", "sub_" + std::to_string(i % 8)},
                           {"region", i % 3 == 0   ? "us"
                                      : i % 3 == 1 ? "eu"
                                                   : "apac"}};
        row[field_embedding] = randomVector();
        rows.emplace_back(std::move(row));
    }

    milvus::InsertResponse insert_response;
    status = client->Insert(milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)),
                            insert_response);
    util::CheckStatus("insert", status);

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
case1SingleField(milvus::MilvusClientV2Ptr& client) {
    auto top_hits = std::make_shared<milvus::AggregationTopHits>(3);
    auto aggregation = std::make_shared<milvus::SearchAggregation>(std::vector<std::string>{field_brand}, 4);
    aggregation->WithTopHits(top_hits);

    auto request = milvus::SearchRequest()
                       .WithCollectionName(collection_name)
                       .WithAnnsField(field_embedding)
                       .WithLimit(search_limit)
                       .AddOutputField(field_id)
                       .AddOutputField(field_brand)
                       .AddOutputField(field_price)
                       .WithSearchAggregation(aggregation);
    addQueryVectors(request);
    executeSearch(client, "Case 1: single field grouping", request);
}

void
case2CompositeKeyWithMetrics(milvus::MilvusClientV2Ptr& client) {
    auto top_hits = std::make_shared<milvus::AggregationTopHits>(2);
    top_hits->AddSort({field_rating, milvus::AggregationDirection::DESC});
    auto aggregation =
        std::make_shared<milvus::SearchAggregation>(std::vector<std::string>{field_brand, field_color}, 5);
    aggregation->AddMetric("avg_price", {milvus::AggregationMetricOp::AVG, field_price})
        .AddMetric("doc_count", {milvus::AggregationMetricOp::COUNT, "*"})
        .AddOrder({"avg_price", milvus::AggregationDirection::DESC})
        .WithTopHits(top_hits);

    auto request = milvus::SearchRequest()
                       .WithCollectionName(collection_name)
                       .WithAnnsField(field_embedding)
                       .WithLimit(search_limit)
                       .AddOutputField(field_id)
                       .AddOutputField(field_brand)
                       .AddOutputField(field_color)
                       .AddOutputField(field_price)
                       .AddOutputField(field_rating)
                       .WithSearchAggregation(aggregation);
    addQueryVectors(request);
    executeSearch(client, "Case 2: composite key + metrics + ordering", request);
}

void
case3JsonField(milvus::MilvusClientV2Ptr& client) {
    auto top_hits = std::make_shared<milvus::AggregationTopHits>(2);
    auto aggregation = std::make_shared<milvus::SearchAggregation>(std::vector<std::string>{"meta['region']"}, 4);
    aggregation->AddMetric("avg_score", {milvus::AggregationMetricOp::AVG, "_score"})
        .AddOrder({"avg_score", milvus::AggregationDirection::DESC})
        .WithTopHits(top_hits);

    auto request = milvus::SearchRequest()
                       .WithCollectionName(collection_name)
                       .WithAnnsField(field_embedding)
                       .WithLimit(search_limit)
                       .AddOutputField(field_id)
                       .AddOutputField(field_category)
                       .WithSearchAggregation(aggregation);
    addQueryVectors(request);
    executeSearch(client, "Case 3: JSON path grouping", request);
}

void
case4TwoLevelNested(milvus::MilvusClientV2Ptr& client) {
    auto sub_top_hits = std::make_shared<milvus::AggregationTopHits>(2);
    sub_top_hits->AddSort({field_price, milvus::AggregationDirection::ASC});
    auto sub_aggregation = std::make_shared<milvus::SearchAggregation>(std::vector<std::string>{field_brand}, 2);
    sub_aggregation->AddMetric("avg_rating", {milvus::AggregationMetricOp::AVG, field_rating})
        .AddOrder({"avg_rating", milvus::AggregationDirection::DESC})
        .WithTopHits(sub_top_hits);

    auto top_hits = std::make_shared<milvus::AggregationTopHits>(2);
    top_hits->AddSort({"_score", milvus::AggregationDirection::DESC});
    auto aggregation = std::make_shared<milvus::SearchAggregation>(std::vector<std::string>{field_category}, 3);
    aggregation->AddMetric("total_revenue", {milvus::AggregationMetricOp::SUM, field_price})
        .AddMetric("item_count", {milvus::AggregationMetricOp::COUNT, "*"})
        .AddOrder({"total_revenue", milvus::AggregationDirection::DESC})
        .WithTopHits(top_hits)
        .WithSubAggregation(sub_aggregation);

    auto request = milvus::SearchRequest()
                       .WithCollectionName(collection_name)
                       .WithAnnsField(field_embedding)
                       .WithLimit(search_limit)
                       .WithFilter(std::string(field_in_stock) + " == true")
                       .AddOutputField(field_id)
                       .AddOutputField(field_category)
                       .AddOutputField(field_brand)
                       .AddOutputField(field_price)
                       .AddOutputField(field_rating)
                       .WithSearchAggregation(aggregation);
    addQueryVectors(request);
    executeSearch(client, "Case 4: two-level nested", request);
}

void
case5ThreeLevelNested(milvus::MilvusClientV2Ptr& client) {
    auto level3_top_hits = std::make_shared<milvus::AggregationTopHits>(2);
    level3_top_hits->AddSort({field_price, milvus::AggregationDirection::ASC});
    auto level3 = std::make_shared<milvus::SearchAggregation>(std::vector<std::string>{field_sku, field_color}, 3);
    level3->AddMetric("min_price", {milvus::AggregationMetricOp::MIN, field_price})
        .AddMetric("item_count", {milvus::AggregationMetricOp::COUNT, "*"})
        .AddOrder({"min_price", milvus::AggregationDirection::ASC})
        .WithTopHits(level3_top_hits);

    auto level2_top_hits = std::make_shared<milvus::AggregationTopHits>(2);
    level2_top_hits->AddSort({field_rating, milvus::AggregationDirection::DESC});
    auto level2 = std::make_shared<milvus::SearchAggregation>(std::vector<std::string>{field_brand}, 3);
    level2->AddMetric("brand_revenue", {milvus::AggregationMetricOp::SUM, field_price})
        .AddMetric("avg_rating", {milvus::AggregationMetricOp::AVG, field_rating})
        .AddOrder({"brand_revenue", milvus::AggregationDirection::DESC})
        .WithTopHits(level2_top_hits)
        .WithSubAggregation(level3);

    auto level1_top_hits = std::make_shared<milvus::AggregationTopHits>(2);
    level1_top_hits->AddSort({"_score", milvus::AggregationDirection::ASC});
    auto level1 = std::make_shared<milvus::SearchAggregation>(std::vector<std::string>{field_category}, 3);
    level1->AddMetric("total_revenue", {milvus::AggregationMetricOp::SUM, field_price})
        .AddMetric("item_count", {milvus::AggregationMetricOp::COUNT, "*"})
        .AddOrder({"total_revenue", milvus::AggregationDirection::DESC})
        .WithTopHits(level1_top_hits)
        .WithSubAggregation(level2);

    auto request = milvus::SearchRequest()
                       .WithCollectionName(collection_name)
                       .WithAnnsField(field_embedding)
                       .WithMetricType(milvus::MetricType::L2)
                       .WithLimit(search_limit)
                       .AddExtraParam("nprobe", "16")
                       .AddOutputField(field_id)
                       .AddOutputField(field_category)
                       .AddOutputField(field_brand)
                       .AddOutputField(field_sku)
                       .AddOutputField(field_color)
                       .AddOutputField(field_price)
                       .AddOutputField(field_rating)
                       .WithSearchAggregation(level1);
    addQueryVectors(request);
    executeSearch(client, "Case 5: three-level nested", request);
}

}  // namespace

int
main(int argc, char* argv[]) {
    std::cout << "Example start..." << std::endl;

    auto client = milvus::MilvusClientV2::Create();
    milvus::ConnectParam connect_param{"http://localhost:19530"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    buildCollection(client);
    case1SingleField(client);
    case2CompositeKeyWithMetrics(client);
    // JSON/dynamic-field aggregation is not supported by the server yet.
    // case3JsonField(client);
    case4TwoLevelNested(client);
    case5ThreeLevelNested(client);

    client->Disconnect();
    return 0;
}
