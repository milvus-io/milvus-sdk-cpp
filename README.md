# Milvus Cpp SDK

[![license](https://img.shields.io/hexpm/l/plug.svg?color=green)](https://github.com/milvus-io/milvus-sdk/blob/master/LICENSE)
[![Mergify Status][mergify-status]][mergify]

Cpp SDK for [Milvus](https://github.com/milvus-io/milvus).

To contribute to this project, please read our [contribution guidelines](https://github.com/milvus-io/milvus/blob/master/CONTRIBUTING.md) and [Development Guide](DEVELOPMENT.md) first.


## Compatibility

The following collection shows Milvus versions and recommended milvus-cpp-sdk versions:

| Milvus version | Recommended SDK version |
|:-----:|:-----:|
| 2.3.x | 2.3(branch)  |
| 2.4.x | v2.4.1  |
| 2.5.x | v2.5.4  |
| 2.6.x | v2.6.3  |
| 3.0.x | v3.0.2  |


## Installation
### Prerequisites
- C++ compiler with C++14 support (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.14+
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

The project uses [Conan 2](https://conan.io/) to manage dependencies. The `scripts/build.sh` handles Conan integration automatically. You can also use Conan directly:

```shell
conan install . --build=missing -s build_type=Release
cmake --preset conan-release
cmake --build build/Release
```

See [Development Guide](DEVELOPMENT.md) for more details.

## Use milvus-sdk-cpp in your project

### Choose the client API

New applications should use `MilvusClientV2`, which provides the current request/response-style API. The original
`MilvusClient` API is retained for compatibility with existing applications and is in maintenance mode.

### Quick start with MilvusClientV2

The following program connects to Milvus, creates a simple collection, inserts three rows, and performs a vector
search:

```cpp
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "milvus/MilvusClientV2.h"

int
main() {
    auto client = milvus::MilvusClientV2::Create();

    auto status = client->Connect(milvus::ConnectParam("http://localhost:19530"));
    if (!status.IsOk()) {
        std::cerr << "Failed to connect to Milvus: " << status.Message() << std::endl;
        return 1;
    }

    const std::string collection_name = "cpp_quickstart";
    client->DropCollection(milvus::DropCollectionRequest().WithCollectionName(collection_name));

    status = client->CreateCollection(milvus::CreateSimpleCollectionRequest()
                                          .WithCollectionName(collection_name)
                                          .WithDimension(4));
    if (!status.IsOk()) {
        std::cerr << "Failed to create collection: " << status.Message() << std::endl;
        return 1;
    }

    milvus::EntityRows rows = {
        {{"id", 1}, {"vector", std::vector<float>{0.1F, 0.2F, 0.3F, 0.4F}}},
        {{"id", 2}, {"vector", std::vector<float>{0.2F, 0.3F, 0.4F, 0.5F}}},
        {{"id", 3}, {"vector", std::vector<float>{0.3F, 0.4F, 0.5F, 0.6F}}},
    };

    milvus::InsertResponse insert_response;
    status = client->Insert(
        milvus::InsertRequest().WithCollectionName(collection_name).WithRowsData(std::move(rows)), insert_response);
    if (!status.IsOk()) {
        std::cerr << "Failed to insert rows: " << status.Message() << std::endl;
        return 1;
    }

    auto search_request = milvus::SearchRequest()
                              .WithCollectionName(collection_name)
                              .WithLimit(2)
                              .WithAnnsField("vector")
                              .AddFloatVector({0.1F, 0.2F, 0.3F, 0.4F})
                              .WithConsistencyLevel(milvus::ConsistencyLevel::STRONG);

    milvus::SearchResponse search_response;
    status = client->Search(search_request, search_response);
    if (!status.IsOk()) {
        std::cerr << "Failed to search: " << status.Message() << std::endl;
        return 1;
    }

    for (const auto& result : search_response.Results().Results()) {
        const auto ids = result.Ids();
        for (size_t i = 0; i < result.Scores().size(); ++i) {
            std::cout << "id=" << ids.IntIDArray()[i] << ", score=" << result.Scores()[i] << std::endl;
        }
    }

    client->Disconnect();
    return 0;
}
```

This example expects a Milvus server at `http://localhost:19530`.

### Link with CMake

After installing the SDK, consume its exported CMake package and target:

```cmake
cmake_minimum_required(VERSION 3.14)
project(milvus_quickstart LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(milvus_sdk CONFIG REQUIRED)

add_executable(milvus_quickstart main.cpp)
target_link_libraries(milvus_quickstart PRIVATE milvus_sdk::milvus_sdk)
```

If the SDK was installed to a custom prefix, provide it while configuring your project:

```shell
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/install
cmake --build build
```

The same `find_package(milvus_sdk)` call and `milvus_sdk::milvus_sdk` target are generated when consuming the SDK as
a Conan package.

If you want to integrate `milvus-sdk-cpp` into your own C++ application, the recommended starting point is the companion example repository:

- [milvus-sdk-cpp-example](https://github.com/milvus-io/milvus-sdk-cpp-example)

That repository shows three practical integration modes:

- **without-conan**: build the SDK and its dependencies from source with CMake/FetchContent
- **conan-for-dependencies**: build `milvus-sdk-cpp` from source while using Conan for its dependencies
- **conan-managed**: consume `milvus-sdk-cpp` itself as a Conan package

A typical source-install workflow is:

1. Build and install `milvus-sdk-cpp`
2. Point your project's `CMAKE_PREFIX_PATH` to the installation prefix
3. Include SDK headers such as `milvus/MilvusClientV2.h`
4. Link your executable against `milvus_sdk::milvus_sdk`

If you prefer a complete, working reference project instead of a minimal snippet, use the example repository above. It includes ready-to-build `CMakeLists.txt` files and build scripts for the supported integration approaches.


## License
[Apache License 2.0](LICENSE)


[mergify]: https://mergify.io
[mergify-status]: https://img.shields.io/endpoint.svg?url=https://gh.mergify.io/badges/milvus-io/milvus-sdk-cpp&style=plastic
