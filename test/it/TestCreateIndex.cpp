#include <gtest/gtest.h>

#include "mocks/MilvusMockedTest.h"

using ::milvus::StatusCode;
using ::milvus::proto::milvus::CreateIndexRequest;
using ::testing::_;
using ::testing::AllOf;
using ::testing::Property;

TEST_F(MilvusMockedTest, DISABLED_TestCreateIndex) {
    milvus::ConnectParam connect_param{"127.0.0.1", server_.ListenPort()};
    client_->Connect(connect_param);

    std::string collection_name = "test_collection";
    std::string field_name = "test_field";
    std::string index_name = "test_index";
    std::string db_name = "test_db";
    int64_t index_id = 0;
    std::unordered_map<std::string, std::string> params;
    params["nlist"] = "10";
    params["nprobe"] = "10";
    params["nbits"] = "10";
    params["nlevel"] = "10";

    milvus::IndexDesc index_desc(field_name, index_name, index_id, params);
    const auto progress_monitor = ::milvus::ProgressMonitor::NoWait();

    milvus::proto::milvus::CreateIndexRequest request;
    request.set_collection_name(collection_name);
    request.set_db_name(index_name);
    request.set_field_name(field_name);

    auto status = client_->CreateIndex(collection_name, index_desc, progress_monitor);
    EXPECT_TRUE(status.IsOk());
}