# Milvus C++ SDK tutorials

Each directory is an independent C++ project. It consumes the published Milvus C++ SDK through Conan, rather than
including this repository's source tree.

Prerequisites: CMake 3.16+, a C++14 compiler, Conan 2.x, and an accessible Milvus server.

## Configure the Conan remote

The tutorials default to `milvus-sdk-cpp/3.0.2@milvus/dev`. This package is hosted on the Milvus
Artifactory remote rather than ConanCenter. Add the remote once before the first tutorial build:

```bash
conan remote add default-conan-local2 \
  https://milvus01.jfrog.io/artifactory/api/conan/default-conan-local2 \
  --force --allowed-packages "milvus-sdk-cpp/*"
```

Override the package reference when needed:

```bash
MILVUS_SDK_VERSION=3.0.3 MILVUS_SDK_USER=myuser MILVUS_SDK_CHANNEL=stable make
```

Build and run one tutorial from its directory:

```bash
cd tutorial/1_quickstart
make
make run
```

## Start here

If this is your first Milvus C++ SDK program, start with the quickstart:

```bash
make -C tutorial/1_quickstart
make -C tutorial/1_quickstart run
```

It expects Milvus at `http://localhost:19530` with the default `root:Milvus` credentials. For a
different server, set `MILVUS_URI` and `MILVUS_TOKEN`:

```bash
MILVUS_URI="https://your-milvus-endpoint" \
MILVUS_TOKEN="your-token" \
make -C tutorial/1_quickstart run
```

The source-linked examples used for SDK development remain under [`examples/`](../examples/README.md).

Connection settings default to `MILVUS_URI=http://localhost:19530` and `MILVUS_TOKEN=root:Milvus`.

Repository maintainers can configure and compile all standalone tutorial projects with:

```bash
make tutorials
```

The target first creates a temporary Conan package from the current checkout, then builds all eight
tutorials against that package. The Linux CI workflow runs it to catch packaging, toolchain,
include, and SDK API drift before release.

After building, run one tutorial from the repository root by its short or directory name:

```bash
make run-tutorial quickstart
make run-tutorial 6_dql
```

Tutorials connect to Milvus and create or modify resources. Review the selected tutorial and its
connection settings before running it. The RBAC tutorial also accepts `MILVUS_USER_PASSWORD`.

## Beginner tutorials

- [1_quickstart](1_quickstart/): connect, create a collection, insert data, search, and clean up.
- [2_collection](2_collection/): collection schema and lifecycle.
- [3_schema](3_schema/): scalar, text, and vector field definitions.
- [4_index](4_index/): vector and scalar indexes.
- [5_dml](5_dml/): insert, upsert, delete, and verification.
- [6_dql](6_dql/): query, search, hybrid search, and iterators.

## Advanced tutorials

- [7_database](7_database/): database lifecycle and properties.
- [8_rbac](8_rbac/): users, roles, privilege groups, and grants.

## Common troubleshooting

- `connection refused`: start Milvus or set `MILVUS_URI` to a reachable endpoint.
- `unauthenticated` or `permission denied`: set a valid `MILVUS_TOKEN` with the required privileges.
- Conan cannot find the SDK package: confirm that `default-conan-local2` is enabled with
  `conan remote list`, then check `MILVUS_SDK_VERSION`, `MILVUS_SDK_USER`, and
  `MILVUS_SDK_CHANNEL` before running `make clean` and `make` again.
- load or index timeout: verify that Milvus is healthy and has enough CPU and memory.
- RBAC failures: use an administrator token and ensure authorization is enabled for the RBAC tutorial.
