# Tutorial 6: Query and search data

This project creates a collection with dense and sparse vector indexes, inserts sample documents,
and demonstrates the main V2 data-query interfaces:

1. `Query` for scalar filtering and selected output fields.
2. `Search` for nearest-neighbor search on a dense vector field.
3. `HybridSearch` for weighted dense+sparse reranking.
4. `QueryIterator` for bounded pages of filtered entities.
5. `SearchIterator` for paged nearest-neighbor results.

The source prints returned rows after each operation. Iterator limits and batch sizes are deliberately
small so pagination is easy to observe.

## Prerequisites and run

You need C++14, CMake 3.16+, Python 3, Conan 2, and Milvus 2.6 or later. Defaults are
`MILVUS_URI=http://localhost:19530` and `MILVUS_TOKEN=root:Milvus`.

```bash
make
make run
```

Set `MILVUS_URI` and `MILVUS_TOKEN` for another server. The tutorial uses
`milvus-sdk-cpp/3.0.2@milvus/dev` unless the `MILVUS_SDK_*` variables override it, and drops its
temporary collection before exiting.

## Expected output

Scores and ordering depend on the server, but the operation sequence includes:

```text
Calling Connect...
Connect succeeded.
Calling DropCollection for stale data...
Stale collection cleanup completed.
Calling CreateCollection...
CreateCollection succeeded.
Calling LoadCollection...
LoadCollection succeeded.
Calling Insert...
Insert succeeded.
Inserted rows: 12
Calling Query...
Query succeeded.
Query results:
  ...
Calling Search...
Search succeeded.
Search results:
  ...
Calling HybridSearch...
HybridSearch succeeded.
Hybrid search results:
  ...
Calling QueryIterator...
QueryIterator succeeded.
Calling QueryIterator::Next...
QueryIterator::Next succeeded with ... rows.
Calling SearchIterator...
SearchIterator succeeded.
Calling SearchIterator::Next...
SearchIterator::Next succeeded with ... rows.
Calling ReleaseCollection...
ReleaseCollection succeeded.
Calling DropCollection...
DropCollection succeeded.
```

## Troubleshooting

- query/search errors: verify that the collection is loaded and requested fields/indexes exist.
- hybrid search errors: check dense and sparse index and metric compatibility.
- an iterator may return a partial page and then an empty final page.
- connection or authentication errors: check `MILVUS_URI` and `MILVUS_TOKEN`.
