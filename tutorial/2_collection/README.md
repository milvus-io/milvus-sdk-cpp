# Tutorial 2: Manage collections

This standalone C++ project demonstrates the collection lifecycle with `MilvusClientV2`.
It creates a collection from a schema, checks and describes the collection, changes properties,
loads it, reads state and statistics, releases it, renames it, truncates its data, and drops it.
An `AutoIndex` is included only so Milvus can load the vector collection; Tutorial 4 covers index
creation and administration in detail.

## API flow

1. `Connect` authenticates the client.
2. `CreateCollection` stores the schema, adds the load prerequisite `AutoIndex`, and sets bounded consistency.
3. `HasCollection` and `DescribeCollection` inspect existence and metadata.
4. `AlterCollectionProperties` and `DropCollectionProperties` manage TTL configuration.
5. `LoadCollection`, `GetLoadState`, and `GetCollectionStats` inspect serving state.
6. `ReleaseCollection`, `RenameCollection`, `TruncateCollection`, and `DropCollection` finish cleanup.

## Prerequisites and configuration

Install a C++14 toolchain, CMake 3.16+, Python 3, and Conan 2. Run Milvus 2.6 or later.
The defaults are `MILVUS_URI=http://localhost:19530` and `MILVUS_TOKEN=root:Milvus`.

```bash
MILVUS_URI="https://your-endpoint" MILVUS_TOKEN="your-token" make run
```

## Build and run

```bash
make
make run
```

The project consumes `milvus-sdk-cpp/3.0.2@milvus/dev` by default. Override the package with
`MILVUS_SDK_VERSION`, `MILVUS_SDK_USER`, and `MILVUS_SDK_CHANNEL`. The tutorial removes its
temporary collection; use `make clean` to remove build outputs.

## Expected output

```text
Calling Connect...
Connect succeeded.
Calling DropCollection for stale collection CPP_TUTORIAL_COLLECTION...
DropCollection completed for CPP_TUTORIAL_COLLECTION.
Calling DropCollection for stale collection CPP_TUTORIAL_COLLECTION_RENAMED...
DropCollection completed for CPP_TUTORIAL_COLLECTION_RENAMED.
Calling CreateCollection...
CreateCollection succeeded.
Calling HasCollection...
HasCollection succeeded.
Collection exists: 1
Calling DescribeCollection...
DescribeCollection succeeded.
Collection ID: ...
Calling AlterCollectionProperties...
AlterCollectionProperties succeeded.
Calling DescribeCollection again...
DescribeCollection succeeded.
TTL: 3600
Calling DropCollectionProperties...
DropCollectionProperties succeeded.
Calling LoadCollection...
LoadCollection succeeded.
Calling GetLoadState...
GetLoadState succeeded.
Load state: ..., progress=100%
Calling GetCollectionStats...
GetCollectionStats succeeded.
Row count: 0
Calling ReleaseCollection...
ReleaseCollection succeeded.
Calling RenameCollection...
RenameCollection succeeded.
Calling HasCollection for CPP_TUTORIAL_COLLECTION_RENAMED...
HasCollection completed for CPP_TUTORIAL_COLLECTION_RENAMED.
Renamed collection exists: 1
Calling TruncateCollection...
TruncateCollection succeeded.
Calling DropCollection...
DropCollection succeeded.
```

## Troubleshooting

- collection administration errors: use a token allowed to create and alter collections.
- load timeout: check server health and available resources.
- rerunning after interruption: the tutorial removes its fixed collection names first.
- connection or authentication errors: check `MILVUS_URI` and `MILVUS_TOKEN`.
