
#include "MilvusMockedTest.h"

void
milvus::MilvusMockedTest::SetUp() {
    server_.Start();
    client_ = milvus::MilvusClient::Create();
}

void
milvus::MilvusMockedTest::TearDown() {
}