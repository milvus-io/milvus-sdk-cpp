#include <cstdlib>
#include <iostream>
#include <string>

#include "milvus/MilvusClientV2.h"

namespace {
const char*
Env(const char* n, const char* d) {
    const char* v = std::getenv(n);
    return v ? v : d;
}
bool
Ok(const milvus::Status& s, const std::string& op) {
    if (s.IsOk()) {
        return true;
    }
    std::cerr << "Failed to " << op << ": " << s.Message() << std::endl;
    return false;
}
bool
PrintDatabases(milvus::MilvusClientV2Ptr& client, const std::string& heading) {
    milvus::ListDatabasesResponse response;
    // ListDatabases returns every database name visible to the authenticated user.
    std::cout << "Calling ListDatabases..." << std::endl;
    auto status = client->ListDatabases(milvus::ListDatabasesRequest(), response);
    if (!Ok(status, "list databases")) {
        return false;
    }
    std::cout << "ListDatabases succeeded." << std::endl;
    std::cout << heading << ": ";
    for (const auto& name : response.DatabaseNames()) {
        std::cout << name << " ";
    }
    std::cout << std::endl;
    return true;
}
}  // namespace

int
main() {
    auto client = milvus::MilvusClientV2::Create();
    // Connect authenticates with the configured Milvus endpoint for database administration.
    std::cout << "Calling Connect..." << std::endl;
    auto status = client->Connect(
        milvus::ConnectParam{Env("MILVUS_URI", "http://localhost:19530"), Env("MILVUS_TOKEN", "root:Milvus")});
    if (!Ok(status, "connect")) {
        return 1;
    }
    std::cout << "Connect succeeded." << std::endl;

    const std::string database = "CPP_TUTORIAL_DATABASE";
    if (!PrintDatabases(client, "Databases before tutorial")) {
        return 1;
    }

    std::cout << "Calling UseDatabase for default..." << std::endl;
    status = client->UseDatabase("default");
    if (!Ok(status, "use default database")) {
        return 1;
    }
    std::cout << "UseDatabase succeeded." << std::endl;
    std::cout << "Calling DropDatabase for stale data..." << std::endl;
    status = client->DropDatabase(milvus::DropDatabaseRequest().WithDatabaseName(database));
    if (!Ok(status, "drop stale database")) {
        return 1;
    }
    std::cout << "Stale database cleanup completed." << std::endl;

    // CreateDatabase creates a logical database. Its name is used by later requests and by
    // UseDatabase.
    std::cout << "Calling CreateDatabase..." << std::endl;
    status = client->CreateDatabase(milvus::CreateDatabaseRequest().WithDatabaseName(database));
    if (!Ok(status, "create database")) {
        return 1;
    }
    std::cout << "CreateDatabase succeeded." << std::endl;

    milvus::DescribeDatabaseResponse described;
    // DescribeDatabase returns metadata and properties for the named database.
    std::cout << "Calling DescribeDatabase..." << std::endl;
    status = client->DescribeDatabase(milvus::DescribeDatabaseRequest().WithDatabaseName(database), described);
    if (!Ok(status, "describe database")) {
        return 1;
    }
    std::cout << "DescribeDatabase succeeded." << std::endl;
    std::cout << "Created database: " << described.Desc().Name() << std::endl;

    // AlterDatabaseProperties adds or replaces database property key/value pairs. This setting
    // configures the default replica count for collections in the database.
    std::cout << "Calling AlterDatabaseProperties..." << std::endl;
    status =
        client->AlterDatabaseProperties(milvus::AlterDatabasePropertiesRequest().WithDatabaseName(database).AddProperty(
            "database.replica.number", "1"));
    if (!Ok(status, "alter database properties")) {
        return 1;
    }
    std::cout << "AlterDatabaseProperties succeeded." << std::endl;

    // DescribeDatabase is called again to read back the updated property map.
    std::cout << "Calling DescribeDatabase again..." << std::endl;
    status = client->DescribeDatabase(milvus::DescribeDatabaseRequest().WithDatabaseName(database), described);
    if (!Ok(status, "describe database properties")) {
        return 1;
    }
    std::cout << "DescribeDatabase succeeded." << std::endl;
    std::cout << "database.replica.number: " << described.Desc().Properties().at("database.replica.number")
              << std::endl;

    // DropDatabaseProperties removes the specified key while preserving other properties.
    std::cout << "Calling DropDatabaseProperties..." << std::endl;
    status = client->DropDatabaseProperties(
        milvus::DropDatabasePropertiesRequest().WithDatabaseName(database).AddPropertyKey("database.replica.number"));
    if (!Ok(status, "drop database property")) {
        return 1;
    }
    std::cout << "DropDatabaseProperties succeeded." << std::endl;

    // DescribeDatabase reads the property map again to verify that the requested key was removed.
    std::cout << "Calling DescribeDatabase to verify property removal..." << std::endl;
    status = client->DescribeDatabase(milvus::DescribeDatabaseRequest().WithDatabaseName(database), described);
    if (!Ok(status, "verify database property removal")) {
        return 1;
    }
    std::cout << "DescribeDatabase succeeded." << std::endl;
    if (described.Desc().Properties().count("database.replica.number") != 0) {
        std::cerr << "Failed to verify database property removal: database.replica.number is still present"
                  << std::endl;
        return 1;
    }
    std::cout << "Verified database.replica.number was removed." << std::endl;

    // UseDatabase selects this database for requests that omit an explicit database name.
    std::cout << "Calling UseDatabase for " << database << "..." << std::endl;
    status = client->UseDatabase(database);
    if (!Ok(status, "use database")) {
        return 1;
    }
    std::cout << "UseDatabase succeeded." << std::endl;
    std::string current_database;
    std::cout << "Calling CurrentUsedDatabase..." << std::endl;
    status = client->CurrentUsedDatabase(current_database);
    if (!Ok(status, "get current database")) {
        return 1;
    }
    std::cout << "CurrentUsedDatabase succeeded." << std::endl;
    std::cout << "Selected database: " << current_database << std::endl;

    // Return to the default database because the currently selected database cannot be dropped.
    std::cout << "Calling UseDatabase for default..." << std::endl;
    status = client->UseDatabase("default");
    if (!Ok(status, "use default database")) {
        return 1;
    }
    std::cout << "UseDatabase succeeded." << std::endl;
    // DropDatabase permanently removes the named database; it must contain no collections.
    std::cout << "Calling DropDatabase..." << std::endl;
    status = client->DropDatabase(milvus::DropDatabaseRequest().WithDatabaseName(database));
    if (!Ok(status, "drop database")) {
        return 1;
    }
    std::cout << "DropDatabase succeeded." << std::endl;
    if (!PrintDatabases(client, "Databases after cleanup")) {
        return 1;
    }

    return 0;
}
