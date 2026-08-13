# Tutorial 5: Insert, upsert, and delete data

This tutorial demonstrates the V2 data-manipulation APIs against a small loaded collection.

1. Insert row-oriented entities with `EntityRows`.
2. Insert the same kind of data using column-oriented `FieldData`.
3. Fully upsert an entity, replacing all fields.
4. Partially upsert an entity, changing only supplied fields.
5. Delete by primary-key IDs and by a boolean filter expression.
6. Release and drop the tutorial collection.

An insert uses either rows or columns. All columns in a request must have the same non-zero row
count. Upserts require a primary key; partial upserts may omit unchanged non-primary fields.

## Prerequisites and run

Install C++14, CMake 3.16+, Python 3, and Conan 2. Run Milvus 2.6 or later. The defaults are
`MILVUS_URI=http://localhost:19530` and `MILVUS_TOKEN=root:Milvus`.

```bash
make
make run
```

The default package is `milvus-sdk-cpp/3.0.2@milvus/dev`; override it with `MILVUS_SDK_VERSION`,
`MILVUS_SDK_USER`, and `MILVUS_SDK_CHANNEL`. `make clean` removes build output.

## Expected output

```text
Calling Connect...
Connect succeeded.
Calling DropCollection for stale data...
Stale collection cleanup completed.
Calling CreateCollection...
CreateCollection succeeded.
Calling LoadCollection...
LoadCollection succeeded.
Calling Insert with row data...
Insert with row data succeeded.
Inserted rows: 2
Calling Insert with column data...
Insert with column data succeeded.
Inserted column rows: 2
Calling Upsert for a full row...
Full Upsert succeeded.
Upserted rows: 1
Calling Upsert for a partial update...
Partial Upsert succeeded.
Calling Delete with primary-key IDs...
Delete by IDs succeeded.
Calling Delete with a filter...
Delete by filter succeeded.
Deleted by filter: ...
Calling ReleaseCollection...
ReleaseCollection succeeded.
Calling DropCollection...
DropCollection succeeded.
```

## Troubleshooting

- insert or upsert validation errors: match field names/types and keep column lengths equal.
- delete visibility depends on consistency; use strong consistency when immediately verifying mutations.
- connection or authentication errors: check `MILVUS_URI` and `MILVUS_TOKEN`.
- rerunning after interruption: the fixed tutorial collection is removed first.
