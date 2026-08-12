# Tutorial 4: Create and manage indexes

This standalone project creates one vector index and two scalar indexes with `MilvusClientV2`:
HNSW with cosine distance for a float vector, `INVERTED` for text, and `STL_SORT` for a numeric field.

## API flow

1. Create a collection without indexes so index creation is explicit.
2. Call `CreateIndex` with field, index name, type, metric, and HNSW build parameters.
3. Use `ListIndexes` to inspect all indexes and `DescribeIndex` for detailed metadata.
4. Drop one index, then drop the collection and remaining indexes.

Vector indexes require a compatible metric. Scalar choices include `INVERTED` for general filtering
and `STL_SORT` for ordered numeric values; `AutoIndex` is a convenient server-selected alternative.

## Prerequisites and run

Use a C++14 compiler, CMake 3.16+, Python 3, Conan 2, and Milvus 2.6 or later. Defaults are
`MILVUS_URI=http://localhost:19530` and `MILVUS_TOKEN=root:Milvus`.

```bash
make
make run
```

The project consumes `milvus-sdk-cpp/3.0.2@milvus/dev` by default. Package and connection settings
can be overridden with the `MILVUS_SDK_*`, `MILVUS_URI`, and `MILVUS_TOKEN` environment variables.

## Expected output

Index listing order and numeric type values may vary by server, but a successful run includes:

```text
Calling Connect...
Connect succeeded.
Calling DropCollection for stale data...
Stale collection cleanup completed.
Calling CreateCollection...
CreateCollection succeeded.
Calling CreateIndex...
CreateIndex succeeded.
Calling ListIndexes...
ListIndexes succeeded.
embedding_hnsw_idx field=embedding type=...
category_inverted_idx field=category type=...
price_sort_idx field=price type=...
Calling DescribeIndex...
DescribeIndex succeeded.
Described vector indexes: 1
Calling DropIndex...
DropIndex succeeded.
Calling DropCollection...
DropCollection succeeded.
```

## Troubleshooting

- index creation errors: verify field, index type, metric, and extra parameters are compatible.
- index timeout: check Milvus health and available CPU and memory.
- connection or authentication errors: check `MILVUS_URI` and `MILVUS_TOKEN`.
- rerunning after interruption: the fixed tutorial collection is removed first.
