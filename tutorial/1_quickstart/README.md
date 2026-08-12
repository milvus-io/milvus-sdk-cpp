# Tutorial 1: Quick start

This is the smallest complete `MilvusClientV2` application in the C++ SDK. It connects to Milvus,
creates a collection with an integer primary key, text, and a float vector, inserts two rows,
loads the collection, performs a cosine vector search, and removes the collection.

## What you learn

1. Create and connect a `MilvusClientV2` instance.
2. Define a collection schema and an `AutoIndex`.
3. Insert row-oriented `EntityRows`.
4. Load a collection and run a strongly consistent search.
5. Release and drop resources during cleanup.

## Prerequisites

- C++14 compiler, CMake 3.16+, Python 3, and Conan 2.
- Milvus 2.6 or later running at an accessible endpoint.

The tutorial defaults to `MILVUS_URI=http://localhost:19530` and
`MILVUS_TOKEN=root:Milvus`. Override them when needed:

```bash
MILVUS_URI="https://your-endpoint" MILVUS_TOKEN="your-token" make run
```

## Build and run

From this directory:

```bash
make
make run
```

`make` installs `milvus-sdk-cpp` as a Conan dependency (default `3.0.2@milvus/dev`), configures
CMake, and builds `cmake_build/tutorial_quickstart`. Set `MILVUS_SDK_VERSION`,
`MILVUS_SDK_USER`, or `MILVUS_SDK_CHANNEL` to consume another package reference.

The program cleans up its tutorial collection on normal completion. `make clean` removes the local
build directory.

## Expected output

A successful run includes:

```text
Calling Connect...
Connect succeeded.
Calling DropCollection for stale data...
Stale collection cleanup completed.
Calling CreateCollection...
CreateCollection succeeded.
Calling Insert...
Insert succeeded: 2 rows.
Calling LoadCollection...
LoadCollection succeeded.
Calling Search...
Search succeeded.
Search result sets: ...
Calling ReleaseCollection...
ReleaseCollection succeeded.
Calling DropCollection...
DropCollection succeeded.
```

## Troubleshooting

- `connection refused`: start Milvus or set `MILVUS_URI` to the reachable endpoint.
- `unauthenticated` or `permission denied`: set a valid `MILVUS_TOKEN`.
- load or search failures: verify that Milvus is healthy and has enough resources.
