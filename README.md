# Milvus C++ SDK

[![license](https://img.shields.io/hexpm/l/plug.svg?color=green)](LICENSE)
[![Mergify Status][mergify-status]][mergify]

The C++ SDK for [Milvus](https://github.com/milvus-io/milvus).

To contribute, read the [contribution guidelines](https://github.com/milvus-io/milvus/blob/master/CONTRIBUTING.md)
and the [Development Guide](DEVELOPMENT.md).

## Compatibility

The following table lists the recommended Milvus C++ SDK version for each Milvus release line:

| Milvus version | Recommended SDK version |
|:---:|:---:|
| 2.3.x | `2.3` branch |
| 2.4.x | `v2.4.1` |
| 2.5.x | `v2.5.4` |
| 2.6.x | `v2.6.7` |

## Build from source

These commands build the SDK repository from source. If you want to consume the SDK from an
application, see [Use the SDK in your project](#use-the-sdk-in-your-project) or start with the
[tutorials](#tutorials).

### Prerequisites

- C++14-capable compiler: GCC 9.4+ on Linux (GCC 9.4.0 is the oldest version currently tested in CI),
  Clang 5+, or MSVC 2017+
- CMake 3.16+
- Python 3 with pip (for Conan and build tools)

### Quick start

```shell
git clone https://github.com/milvus-io/milvus-sdk-cpp.git
cd milvus-sdk-cpp
bash scripts/install_deps.sh
make
```

### Build and run tests

```shell
make test          # unit tests + integration tests
make st            # system tests (requires Docker)
make coverage      # code coverage report
```

### Install the SDK

```shell
make install       # install to /usr/local
```

Or specify a custom install prefix:

```shell
make install CMAKE_INSTALL_PREFIX=/path/to/install
```

### Build with Conan

The project uses [Conan 2](https://conan.io/) to manage dependencies. `scripts/build.sh` handles
Conan integration automatically. You can also invoke Conan and CMake directly:

```shell
conan install . --build=missing -s build_type=Release
cmake --preset conan-release
cmake --build build/Release
```

See the [Development Guide](DEVELOPMENT.md) for details.

## Use the SDK in your project

### Choose the client API

New applications should use `MilvusClientV2`, which provides the current request/response-style
API. The original `MilvusClient` API is retained for compatibility with existing applications and
is in maintenance mode.

### Link with CMake

After installing the SDK, consume its exported CMake package and target:

```cmake
cmake_minimum_required(VERSION 3.16)
project(milvus_quickstart LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(milvus_sdk CONFIG REQUIRED)

add_executable(milvus_quickstart main.cpp)
target_link_libraries(milvus_quickstart PRIVATE milvus_sdk::milvus_sdk)
```

If the SDK was installed to a custom prefix, provide it when configuring your project:

```shell
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/install
cmake --build build
```

The same `find_package(milvus_sdk)` call and `milvus_sdk::milvus_sdk` target are available when
consuming the SDK as a Conan package.

For complete integration projects, see
[milvus-sdk-cpp-example](https://github.com/milvus-io/milvus-sdk-cpp-example). It demonstrates
three supported approaches:

- **without-conan**: build the SDK and its dependencies from source with CMake/FetchContent
- **conan-for-dependencies**: build the SDK from source while Conan provides its dependencies
- **conan-managed**: consume the SDK itself as a Conan package

### Minimal MilvusClientV2 program

The following program connects to Milvus and then closes the connection:

```cpp
#include <iostream>

#include "milvus/MilvusClientV2.h"

int
main() {
    auto client = milvus::MilvusClientV2::Create();

    auto status = client->Connect(milvus::ConnectParam("http://localhost:19530"));
    if (!status.IsOk()) {
        std::cerr << "Failed to connect to Milvus: " << status.Message() << std::endl;
        return 1;
    }

    client->Disconnect();
    return 0;
}
```

This program expects Milvus at `http://localhost:19530`. For an independently buildable project
that covers collection creation, insertion, loading, search, and cleanup, follow
[Tutorial 1: Quickstart](tutorial/1_quickstart/).

## Tutorials

Standalone CMake/Conan projects are available under [`tutorial`](tutorial/README.md). They consume
the published C++ SDK as an application dependency and provide a beginner-to-advanced sequence.

For a first run, start with the [quickstart](tutorial/1_quickstart/). It demonstrates the complete
connect, create, insert, load, search, and cleanup flow:

1. [`Quickstart`](tutorial/1_quickstart/)
2. [`Collections`](tutorial/2_collection/)
3. [`Schemas`](tutorial/3_schema/)
4. [`Indexes`](tutorial/4_index/)
5. [`DML`](tutorial/5_dml/)
6. [`DQL`](tutorial/6_dql/)
7. [`Databases (advanced)`](tutorial/7_database/)
8. [`RBAC (advanced)`](tutorial/8_rbac/)

Each tutorial is an independent project with its own `CMakeLists.txt`, `conanfile.py`, and
`Makefile`. Its README contains prerequisites, configuration, build/run commands, expected output,
and troubleshooting guidance. See the [`tutorial index`](tutorial/README.md) for shared Conan and
connection settings.

Repository maintainers can build all tutorials against the current checkout and run one without
changing directories:

```shell
make tutorials
make run-tutorial quickstart
```

## More examples

This repository includes additional runnable examples under [`examples`](examples/README.md). New
applications should start with the `MilvusClientV2` examples in `examples/src/v2`.

Build an example and run it from the repository root using its source basename:

```shell
cmake --build cmake_build --target sdk_simple_v2 -j4
make run simple
```

`make run simple` runs `cmake_build/examples/v2/sdk_simple_v2`. Replace `simple` with another V2
example name such as `general`, `dml`, or `hybrid_search`. Examples connect to Milvus and may
create, modify, or remove data; review the selected example and its connection settings first.

## License

[Apache License 2.0](LICENSE)


[mergify]: https://mergify.io
[mergify-status]: https://img.shields.io/endpoint.svg?url=https://gh.mergify.io/badges/milvus-io/milvus-sdk-cpp&style=plastic
