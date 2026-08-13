# Tutorial 7: Manage databases

Databases provide logical namespaces for collections. This standalone C++ project demonstrates:

1. `ListDatabases` to inspect visible databases.
2. `CreateDatabase` and `DescribeDatabase` for database metadata.
3. `AlterDatabaseProperties` and `DropDatabaseProperties` for configuration.
4. `UseDatabase` and `CurrentUsedDatabase` to select and inspect the current database.
5. Returning to `default`, dropping the temporary database, and listing databases to verify cleanup.

The example sets `database.replica.number=1`, reads it back, removes it, and verifies the property
change before cleanup.

## Prerequisites and run

Install C++14, CMake 3.16+, Python 3, and Conan 2. Run Milvus 2.6 or later with credentials that
can administer databases. Defaults are `MILVUS_URI=http://localhost:19530` and `MILVUS_TOKEN=root:Milvus`.

```bash
make
make run
```

Override connection variables or the default Conan package (`milvus-sdk-cpp/3.0.2@milvus/dev`)
with `MILVUS_URI`, `MILVUS_TOKEN`, and `MILVUS_SDK_*`. The selected database is reset to `default`
before it is dropped.

## Expected output

```text
Calling Connect...
Connect succeeded.
Calling ListDatabases...
ListDatabases succeeded.
Databases before tutorial: ...
Calling UseDatabase for default...
UseDatabase succeeded.
Calling DropDatabase for stale data...
Stale database cleanup completed.
Calling CreateDatabase...
CreateDatabase succeeded.
Calling DescribeDatabase...
DescribeDatabase succeeded.
Created database: CPP_TUTORIAL_DATABASE
Calling AlterDatabaseProperties...
AlterDatabaseProperties succeeded.
Calling DescribeDatabase again...
DescribeDatabase succeeded.
database.replica.number: 1
Calling DropDatabaseProperties...
DropDatabaseProperties succeeded.
Calling DescribeDatabase to verify property removal...
DescribeDatabase succeeded.
Verified database.replica.number was removed.
Calling UseDatabase for CPP_TUTORIAL_DATABASE...
UseDatabase succeeded.
Calling CurrentUsedDatabase...
CurrentUsedDatabase succeeded.
Selected database: CPP_TUTORIAL_DATABASE
Calling UseDatabase for default...
UseDatabase succeeded.
Calling DropDatabase...
DropDatabase succeeded.
Calling ListDatabases...
ListDatabases succeeded.
Databases after cleanup: ...
```

## Troubleshooting

- database administration errors: use a token with database-management privileges.
- a selected database cannot be dropped; return to `default` first.
- property errors: verify the server supports `database.replica.number`.
- connection or authentication errors: check `MILVUS_URI` and `MILVUS_TOKEN`.
