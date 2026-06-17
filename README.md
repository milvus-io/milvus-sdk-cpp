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
| 3.0.x | v3.0.0  |


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

If you want to integrate `milvus-sdk-cpp` into your own C++ application, the recommended starting point is the companion example repository:

- [milvus-sdk-cpp-example](https://github.com/milvus-io/milvus-sdk-cpp-example)

That repository shows three practical integration modes:

- **without-conan**: build the SDK and its dependencies from source with CMake/FetchContent
- **conan-for-dependencies**: build `milvus-sdk-cpp` from source while using Conan for its dependencies
- **conan-managed**: consume `milvus-sdk-cpp` itself as a Conan package

A typical workflow is:

1. Build and install `milvus-sdk-cpp`
2. Point your project’s CMake build to the installed headers and library
3. Include SDK headers such as `milvus/MilvusClientV2.h`
4. Link your executable against `milvus_sdk`

For example, after installing the SDK to a custom prefix:

```shell
make install CMAKE_INSTALL_PREFIX=/path/to/install
```

your CMake project can add the SDK include and library paths explicitly:

```cmake
include_directories(/path/to/install/include)
link_directories(/path/to/install/lib)

target_link_libraries(my_program milvus_sdk)
```

If you prefer a complete, working reference project instead of a minimal snippet, use the example repository above. It includes ready-to-build `CMakeLists.txt` files and build scripts for the supported integration approaches.


## License
[Apache License 2.0](LICENSE)


[mergify]: https://mergify.io
[mergify-status]: https://img.shields.io/endpoint.svg?url=https://gh.mergify.io/badges/milvus-io/milvus-sdk-cpp&style=plastic
