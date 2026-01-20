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
#include <string>
#include <thread>

#include "ExampleUtils.h"
#include "milvus/MilvusClientV2.h"

namespace {

// ConvertToBinaryVector and ConvertToBoolArray must follow the same order to organize the bool array
// the 0th bool value is stored at the 0th bit, the 7th bool value is stored at the 7th bit
std::vector<uint8_t>
ConvertToBinaryVector(const std::vector<bool>& bools) {
    // ideally, the length of bools must be equal to vector dimension
    // the length of output std::vector<uint8_t> must be dimension/8
    size_t num_bytes = (bools.size() + 7) / 8;
    std::vector<uint8_t> bytes(num_bytes, 0);

    for (size_t i = 0; i < bools.size(); ++i) {
        size_t byte_index = i / 8;
        size_t bit_pos = i % 8;

        if (bools[i]) {
            bytes[byte_index] |= (1U << bit_pos);
        }
    }

    return bytes;
}

// ConvertToBinaryVector and ConvertToBoolArray must follow the same order to organize the bool array
// read the 0th bool value from the 0th bit, read the 7th bool value from the 7th bit
std::vector<bool>
ConvertToBoolArray(const std::vector<uint8_t>& binary) {
    std::vector<bool> bits;
    bits.reserve(binary.size() * 8);
    for (uint8_t byte : binary) {
        for (int i = 0; i < 8; i++) {
            bool bit_is_set = (byte >> i) & 1;
            bits.push_back(bit_is_set);
        }
    }
    return bits;
}

}  // namespace

int
main(int argc, char* argv[]) {
    printf("Example start...\n");

    auto client = milvus::MilvusClientV2::Create();

    milvus::ConnectParam connect_param{"http://localhost:19530", "root:Milvus"};
    auto status = client->Connect(connect_param);
    util::CheckStatus("connect milvus server", status);

    const std::string collection_name = "CPP_V2_BINARY_VECTOR";
    const std::string field_id = "pk";
    const std::string field_vector = "vector";
    const std::string field_text = "text";
    const uint32_t dimension = 128;

    // collection schema, drop and create collection
    milvus::CollectionSchemaPtr collection_schema = std::make_shared<milvus::CollectionSchema>();
    collection_schema->AddField(milvus::FieldSchema(field_id, milvus::DataType::INT64, "", true, false));
    collection_schema->AddField(
        milvus::FieldSchema(field_vector, milvus::DataType::BINARY_VECTOR).WithDimension(dimension));
    collection_schema->AddField(milvus::FieldSchema(field_text, milvus::DataType::VARCHAR).WithMaxLength(1024));

    status = client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));
    status = client->CreateCollection(
        milvus::CreateCollectionRequest().WithCollectionName(collection_name).WithCollectionSchema(collection_schema));
    util::CheckStatus("create collection: " + collection_name, status);

    // create index
    milvus::IndexDesc index_vector(field_vector, "", milvus::IndexType::BIN_IVF_FLAT, milvus::MetricType::HAMMING);
    index_vector.AddExtraParam(milvus::NLIST, "5");
    status = client->CreateIndex(
        milvus::CreateIndexRequest().WithCollectionName(collection_name).AddIndex(std::move(index_vector)));
    util::CheckStatus("create index on vector field", status);

    // tell server prepare to load collection
    status = client->LoadCollection(milvus::LoadCollectionRequest().WithCollectionName(collection_name));
    util::CheckStatus("load collection: " + collection_name, status);

    {
        // insert some rows by column-based
        auto ids = std::vector<int64_t>{10000, 10001};
        auto texts = std::vector<std::string>{"column-based-1", "column-based-2"};
        auto vectors = util::GenerateBinaryVectors(dimension, 2);
        std::vector<milvus::FieldDataPtr> fields_data{
            std::make_shared<milvus::Int64FieldData>(field_id, ids),
            std::make_shared<milvus::VarCharFieldData>(field_text, texts),
            std::make_shared<milvus::BinaryVecFieldData>(field_vector, vectors)};

        milvus::InsertResponse resp_insert;
        status = client->Insert(
            milvus::InsertRequest().WithCollectionName(collection_name).WithColumnsData(std::move(fields_data)),
            resp_insert);
        util::CheckStatus("insert", status);
        std::cout << resp_insert.Results().InsertCount() << " rows inserted by column-based." << std::endl;
    }

    // prepare original vectors
    const int64_t row_count = 10;
    std::vector<std::vector<bool>> bools_array;
    bools_array.reserve(row_count);
    for (auto i = 0; i < row_count; ++i) {
        bools_array.emplace_back(util::RansomBools(dimension));
    }

    milvus::EntityRows rows;
    rows.reserve(row_count);
    {
        // insert some rows
        for (auto i = 0; i < row_count; ++i) {
            milvus::EntityRow row;
            row[field_id] = i;
            row[field_text] = "row-based-" + std::to_string(i);
            row[field_vector] = ConvertToBinaryVector(bools_array.at(i));

            rows.emplace_back(std::move(row));
        }

        milvus::InsertResponse resp_insert;
        milvus::EntityRows rows_copy = rows;  // the rows are used for search later, make a copy here
        status = client->Insert(
            milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows_copy)),
            resp_insert);
        util::CheckStatus("insert", status);
        std::cout << resp_insert.Results().InsertCount() << " rows inserted by row-based." << std::endl;
    }

    auto q_number_1 = util::RandomeValue<int64_t>(0, row_count - 1);
    auto q_number_2 = util::RandomeValue<int64_t>(0, row_count - 1);
    {
        // query some items from the row-based insert data
        auto q_id_1 = rows[q_number_1][field_id].get<int64_t>();
        auto q_id_2 = rows[q_number_2][field_id].get<int64_t>();
        std::string filter = field_id + " in [" + std::to_string(q_id_1) + ", " + std::to_string(q_id_2) + "]";
        std::cout << "Query with filter expression: " << filter << std::endl;

        auto request =
            milvus::QueryRequest()
                .WithCollectionName(collection_name)
                .AddOutputField(field_vector)
                .AddOutputField(field_text)
                .WithFilter(filter)
                // set to strong level so that the query is executed after the inserted data is consumed by server
                .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

        milvus::QueryResponse response;
        status = client->Query(request, response);
        util::CheckStatus("query", status);

        // the result stores data of each field as column-based, OutputRows() convert the data to JSON rows
        milvus::EntityRows output_rows;
        status = response.Results().OutputRows(output_rows);
        util::CheckStatus("get output rows", status);
        std::cout << "Query results:" << std::endl;
        for (const auto& row : output_rows) {
            std::cout << "\tRow: " << row << std::endl;
            auto binary = row[field_vector].get<std::vector<uint8_t>>();
            auto bools = ConvertToBoolArray(binary);
            auto id = row[field_id].get<int64_t>();
            auto original_bools = bools_array.at(id);
            if (!std::equal(bools.begin(), bools.end(), original_bools.begin())) {
                std::cout << "Output vector is not equal to the original!" << std::endl;
                std::cout << "\tOutput vector: ";
                util::PrintList(bools);
                std::cout << "\tOriginal vector: ";
                util::PrintList(original_bools);
                exit(1);
            }
        }
    }

    {
        // do search
        auto q_vector_1 = rows[q_number_1][field_vector];
        auto q_vector_2 = rows[q_number_2][field_vector];
        std::vector<std::vector<uint8_t>> query_vectors = {q_vector_1.get<std::vector<uint8_t>>(),
                                                           q_vector_2.get<std::vector<uint8_t>>()};
        auto request = milvus::SearchRequest()
                           .WithCollectionName(collection_name)
                           .WithLimit(3)
                           .WithAnnsField(field_vector)
                           .AddOutputField(field_vector)
                           .AddOutputField(field_text)
                           .WithBinaryVectors(std::move(query_vectors))
                           .WithConsistencyLevel(milvus::ConsistencyLevel::BOUNDED);

        std::cout << "Searching the ID." << q_number_1 << " binary vector: " << q_vector_1 << std::endl;
        std::cout << "Searching the ID." << q_number_2 << " binary vector: " << q_vector_2 << std::endl;

        milvus::SearchResponse response;
        status = client->Search(request, response);
        util::CheckStatus("search", status);

        for (auto& result : response.Results().Results()) {
            std::cout << "Result of one target vector:" << std::endl;
            milvus::EntityRows output_rows;
            status = result.OutputRows(output_rows);
            util::CheckStatus("get output rows", status);
            for (const auto& row : output_rows) {
                std::cout << "\t" << row << std::endl;
            }
        }
    }

    client->Disconnect();
    return 0;
}
