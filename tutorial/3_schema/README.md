# Tutorial 3: Define collection schemas

A schema defines the fields and validation rules for entities in a collection. This tutorial
creates and describes three collections to demonstrate the C++ SDK's V2 data types without putting
all fields into one oversized schema.

## Schemas demonstrated

| Collection | Fields covered |
|---|---|
| `CPP_TUTORIAL_SCHEMA_SCALAR` | nullable/default scalar fields, text, JSON, geometry, timestamps, arrays, and a required minimal vector |
| `CPP_TUTORIAL_SCHEMA_VECTOR` | float, binary, Float16, and BFloat16 vectors |
| `CPP_TUTORIAL_SCHEMA_STRUCT` | sparse and Int8 vectors plus a struct-array field with nested fields |

Important rules shown in the source include `WithMaxLength` for `VARCHAR`, `WithDimension` for
dense/binary vectors, array element type and capacity, nullable/default values, and struct sub-fields.
Dynamic fields are disabled so undeclared input is rejected.
Milvus requires at least one vector field in a collection, so the scalar-focused schema includes a
two-dimensional `vector` field solely to satisfy that collection rule.

## Prerequisites and run

You need a C++14 compiler, CMake 3.16+, Python 3, Conan 2, and Milvus 2.6 or later. The default
connection is `http://localhost:19530` with token `root:Milvus`; override it with environment variables.

```bash
make
make run
```

The default Conan package is `milvus-sdk-cpp/3.0.2@milvus/dev`. Set `MILVUS_SDK_VERSION`,
`MILVUS_SDK_USER`, or `MILVUS_SDK_CHANNEL` to select another package. Each temporary collection
is described and then dropped. `make clean` removes local build output.

## Expected output

The tutorial prints each schema collection followed by its field types:

```text
Calling Connect...
Connect succeeded.
Calling DropCollection for stale collection CPP_TUTORIAL_SCHEMA_SCALAR...
Stale collection cleanup completed for CPP_TUTORIAL_SCHEMA_SCALAR.
Calling CreateCollection for CPP_TUTORIAL_SCHEMA_SCALAR...
CreateCollection succeeded for CPP_TUTORIAL_SCHEMA_SCALAR.
Calling DescribeCollection for CPP_TUTORIAL_SCHEMA_SCALAR...
DescribeCollection succeeded for CPP_TUTORIAL_SCHEMA_SCALAR.
  id (type=...)
  varchar_value (type=...)
Calling DropCollection for stale collection CPP_TUTORIAL_SCHEMA_VECTOR...
Stale collection cleanup completed for CPP_TUTORIAL_SCHEMA_VECTOR.
Calling CreateCollection for CPP_TUTORIAL_SCHEMA_VECTOR...
CreateCollection succeeded for CPP_TUTORIAL_SCHEMA_VECTOR.
Calling DescribeCollection for CPP_TUTORIAL_SCHEMA_VECTOR...
DescribeCollection succeeded for CPP_TUTORIAL_SCHEMA_VECTOR.
Calling DropCollection for stale collection CPP_TUTORIAL_SCHEMA_STRUCT...
Stale collection cleanup completed for CPP_TUTORIAL_SCHEMA_STRUCT.
Calling CreateCollection for CPP_TUTORIAL_SCHEMA_STRUCT...
CreateCollection succeeded for CPP_TUTORIAL_SCHEMA_STRUCT.
Calling DescribeCollection for CPP_TUTORIAL_SCHEMA_STRUCT...
DescribeCollection succeeded for CPP_TUTORIAL_SCHEMA_STRUCT.
  events (struct, 3 sub-fields)
Calling DropCollection for CPP_TUTORIAL_SCHEMA_SCALAR...
DropCollection succeeded for CPP_TUTORIAL_SCHEMA_SCALAR.
Calling DropCollection for CPP_TUTORIAL_SCHEMA_VECTOR...
DropCollection succeeded for CPP_TUTORIAL_SCHEMA_VECTOR.
Calling DropCollection for CPP_TUTORIAL_SCHEMA_STRUCT...
DropCollection succeeded for CPP_TUTORIAL_SCHEMA_STRUCT.
```

## Troubleshooting

- schema validation errors: check vector dimensions, string lengths, array capacity, and struct fields.
- unsupported data types: use a Milvus version that supports the demonstrated V2 types.
- connection or authentication errors: check `MILVUS_URI` and `MILVUS_TOKEN`.
- rerunning after interruption: each fixed collection is removed before creation.
