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

#include <algorithm>

#include "MilvusServerTest.h"

using milvus::test::MilvusServerTest;

class MilvusServerTestDatabase : public MilvusServerTest {};

TEST_F(MilvusServerTestDatabase, CreateListDropDatabase) {
    std::string db_name = milvus::test::RanName("db_");

    // create database
    auto status = client_->CreateDatabase(milvus::CreateDatabaseRequest().WithDatabaseName(db_name));
    milvus::test::ExpectStatusOK(status);

    // list databases
    milvus::ListDatabasesResponse list_resp;
    status = client_->ListDatabases(milvus::ListDatabasesRequest(), list_resp);
    milvus::test::ExpectStatusOK(status);

    auto& names = list_resp.DatabaseNames();
    EXPECT_NE(std::find(names.begin(), names.end(), db_name), names.end());

    // use the new database
    status = client_->UseDatabase(db_name);
    milvus::test::ExpectStatusOK(status);

    // switch back to default
    status = client_->UseDatabase("default");
    milvus::test::ExpectStatusOK(status);

    // drop database
    status = client_->DropDatabase(milvus::DropDatabaseRequest().WithDatabaseName(db_name));
    milvus::test::ExpectStatusOK(status);

    // verify dropped
    milvus::ListDatabasesResponse list_resp2;
    status = client_->ListDatabases(milvus::ListDatabasesRequest(), list_resp2);
    milvus::test::ExpectStatusOK(status);

    auto& names2 = list_resp2.DatabaseNames();
    EXPECT_EQ(std::find(names2.begin(), names2.end(), db_name), names2.end());
}

TEST_F(MilvusServerTestDatabase, DescribeAndAlterDatabaseProperties) {
    std::string db_name = milvus::test::RanName("db_");

    // create database
    auto status = client_->CreateDatabase(milvus::CreateDatabaseRequest().WithDatabaseName(db_name));
    milvus::test::ExpectStatusOK(status);

    // alter database properties
    status = client_->AlterDatabaseProperties(
        milvus::AlterDatabasePropertiesRequest().WithDatabaseName(db_name).AddProperty("database.replica.number", "1"));
    milvus::test::ExpectStatusOK(status);

    // verify by describing
    milvus::DescribeDatabaseResponse desc_resp;
    status = client_->DescribeDatabase(milvus::DescribeDatabaseRequest().WithDatabaseName(db_name), desc_resp);
    milvus::test::ExpectStatusOK(status);
    auto props = desc_resp.Desc().Properties();
    EXPECT_GE(props.size(), 1);
    EXPECT_TRUE(props.find("database.replica.number") != props.end());
    EXPECT_EQ(props["database.replica.number"], "1");

    // drop database properties
    status = client_->DropDatabaseProperties(
        milvus::DropDatabasePropertiesRequest().WithDatabaseName(db_name).AddPropertyKey("database.replica.number"));
    milvus::test::ExpectStatusOK(status);

    // verify by describing again
    milvus::DescribeDatabaseResponse desc_resp2;
    status = client_->DescribeDatabase(milvus::DescribeDatabaseRequest().WithDatabaseName(db_name), desc_resp2);
    milvus::test::ExpectStatusOK(status);
    props = desc_resp2.Desc().Properties();
    EXPECT_TRUE(props.find("database.replica.number") == props.end());

    // cleanup
    client_->DropDatabase(milvus::DropDatabaseRequest().WithDatabaseName(db_name));
}

TEST_F(MilvusServerTestDatabase, CurrentUsedDatabase) {
    // default database should be "default"
    std::string db_name;
    auto status = client_->CurrentUsedDatabase(db_name);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(db_name, "default");

    // create and switch to a new database
    std::string new_db = milvus::test::RanName("db_");
    status = client_->CreateDatabase(milvus::CreateDatabaseRequest().WithDatabaseName(new_db));
    milvus::test::ExpectStatusOK(status);

    status = client_->UseDatabase(new_db);
    milvus::test::ExpectStatusOK(status);

    // verify current database changed
    std::string current_db;
    status = client_->CurrentUsedDatabase(current_db);
    milvus::test::ExpectStatusOK(status);
    EXPECT_EQ(current_db, new_db);

    // switch back and cleanup
    client_->UseDatabase("default");
    client_->DropDatabase(milvus::DropDatabaseRequest().WithDatabaseName(new_db));
}
