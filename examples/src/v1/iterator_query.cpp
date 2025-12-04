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

#include <iostream>
#include <set>
#include <string>

#include "ExampleUtils.h"
#include "milvus/MilvusClient.h"

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClient::Create();

    milvus::ConnectParam connect_param{"localhost", 19530, "root", "Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    const std::string collection_name = "CPP_V1_ITERATOR_QUERY";
    const std::string field_id = "user_id";
    const std::string field_name = "user_name";
    const std::string field_age = "user_age";
    const std::string field_face = "user_face";
    const uint32_t dimension = 128;

    // collection schema, drop and create collection
    milvus::CollectionSchema collection_schema(collection_name);
    collection_schema.SetEnableDynamicField(true);
    collection_schema.AddField({field_id, milvus::DataType::INT64, "user id", true, false});
    collection_schema.AddField(milvus::FieldSchema(field_name, milvus::DataType::VARCHAR).WithMaxLength(100));
    collection_schema.AddField({field_age, milvus::DataType::INT8});
    collection_schema.AddField(
        milvus::FieldSchema(field_face, milvus::DataType::FLOAT_VECTOR).WithDimension(dimension));

    status = client->DropCollection(collection_name);
    status = client->CreateCollection(collection_schema);
    util::CheckStatus("create collection: " + collection_name, status);

    // create index
    milvus::IndexDesc index_vector(field_face, "", milvus::IndexType::AUTOINDEX, milvus::MetricType::L2);
    status = client->CreateIndex(collection_name, index_vector);
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(collection_name);
    util::CheckStatus("load collection: " + collection_name, status);

    // insert rows with unordered primary keys
    // the pk will be 50000~59999, 10000~19999, 30000~39999, 90000~99999, 0~9999
    std::vector<int64_t> pk_seeds{50000, 10000, 30000, 90000, 0};
    for (auto seed : pk_seeds) {
        milvus::EntityRows rows;
        for (int64_t k = 0; k < 10000; k++) {
            milvus::EntityRow row;
            auto id = seed + k;
            row[field_id] = id;
            row[field_name] = "my name is " + std::to_string(id);
            row[field_age] = k % 100;
            row[field_face] = util::GenerateFloatVector(dimension);
            row["a"] = id;                            // dynamic field "a"
            row["b"] = "b is " + std::to_string(id);  // dynamic field "b"
            rows.emplace_back(std::move(row));
        }

        milvus::DmlResults dml_results;
        status = client->Insert(collection_name, "", rows, dml_results);
        util::CheckStatus("insert", status);
        std::cout << dml_results.InsertCount() << " rows inserted." << std::endl;
    }

    uint64_t row_count = 0;
    {
        // check row count
        milvus::QueryArguments q_count{};
        q_count.SetCollectionName(collection_name);
        q_count.AddOutputField("count(*)");
        q_count.SetConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        milvus::QueryResults count_result{};
        status = client->Query(q_count, count_result);
        util::CheckStatus("query count(*)", status);
        std::cout << "count(*) = " << count_result.GetRowCount() << std::endl;
        row_count = count_result.GetRowCount();
    }

    auto iterationFunc = [&](int64_t batch, int64_t offset, int64_t limit, const std::string& filter) {
        std::cout << "=====================================================" << std::endl;
        std::cout << "Iterate batch: " + std::to_string(batch) << " offset: " + std::to_string(offset)
                  << " limit: " + std::to_string(limit) << " filter: " + filter << std::endl;
        milvus::QueryIteratorArguments arguments;
        arguments.SetCollectionName(collection_name);
        arguments.SetBatchSize(batch);
        arguments.SetOffset(offset);
        arguments.SetLimit(limit);
        arguments.SetFilter(filter);
        arguments.AddOutputField(field_name);
        arguments.AddOutputField(field_age);
        arguments.AddOutputField("a");  // dynamic field

        milvus::QueryIteratorPtr iterator;
        auto status = client->QueryIterator(arguments, iterator);
        util::CheckStatus("get query iterator", status);

        std::set<int64_t> ids;
        int pages = 0;
        uint64_t total_count = 0;
        while (true) {
            milvus::QueryResults batch_results;
            status = iterator->Next(batch_results);
            util::CheckStatus("iterator next batch", status);
            auto batch_count = batch_results.GetRowCount();
            if (batch_count == 0) {
                std::cout << "query iteration finished" << std::endl;
                break;
            }
            pages++;
            total_count += batch_count;

            milvus::EntityRows rows;
            status = batch_results.OutputRows(rows);
            util::CheckStatus("get output rows", status);
            std::cout << "No." << std::to_string(pages) << " page " << std::to_string(rows.size()) << " rows fetched"
                      << std::endl;
            std::cout << "\tthe first row: " << (*rows.begin()).dump() << std::endl;
            std::cout << "\tthe last row: " << (*rows.rbegin()).dump() << std::endl;
            for (const auto& row : rows) {
                // std::cout << row.dump() << std::endl;
                ids.insert(row[field_id].get<int64_t>());
            }
        }

        // verify the number of returned ids is expected
        // only check when filter is empty because filtering is unpredictable
        if (filter.empty()) {
            auto expected_count = limit < (row_count - offset) ? limit : row_count - offset;
            if (limit < 0) {
                expected_count = row_count - offset;
            }
            if (ids.size() != expected_count) {
                std::cout << "Returned row count is unexpected: " << std::to_string(ids.size()) << " returned vs "
                          << "expected " << std::to_string(expected_count) << std::endl;
                exit(1);
            }
        }

        std::cout << "Total fetched rows: " << std::to_string(total_count) << std::endl;
        std::cout << "=====================================================" << std::endl;
    };

    // batch 3000, offset 25000, limit 100000
    iterationFunc(3000, 25000, 100000, "");
    // batch 25, offset 100, limit 80
    iterationFunc(25, 100, 80, "");
    // batch 1000, offset 0, unlimited
    iterationFunc(5000, 0, -1, "");

    // batch 100, offset 0, unlimited, filter "user_age == 8"
    iterationFunc(100, 0, -1, field_age + " == 8");
    // batch 1000, offset 15000, limit 2500, filter "user_age > 30"
    iterationFunc(1000, 15000, 2500, field_age + " > 30");
    // batch 1000, offset 15000, limit 100000, filter "user_age in [30, 40, 50]"
    iterationFunc(1000, 0, 100000, field_age + " in [30, 40, 50]");

    client->Disconnect();
    return 0;
}
