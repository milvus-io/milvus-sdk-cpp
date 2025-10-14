# Development Guide Milvus C++ SDK
This document helps you set up your development environment and run tests for Milvus C++ SDK.
Please file an [issue](https://github.com/milvus-io/milvus-sdk-cpp/issues/new) if you have any questions.

# Supported Platforms
- Linux
- macOS
- Windows

# Build C++ SDK with your Linux

## Setup development environment
Currently, we tested the below platform and compilers for developing Milvus C++ SDK.

| Platform | Version      | Compiler Tested      | Support                       |
| -------- | ------------ | -------------------- | ----------------------------- |
| Linux    | Ubuntu 18.04 | GCC 7.0.0            | Full (Compile, Lint, Testing) |
| Linux    | Ubuntu 20.04 | GCC 9.3.0            | Full (Compile, Lint, Testing) |
| Linux    | Fedora 38/39 | GCC 11.2             | Compile, Testing              |
| Linux    | CentOS 7     | GCC7 (devtoolset-7)  | Compile, Testing              |

### Clone the code

```shell
$ git clone https://github.com/milvus-io/milvus-sdk-cpp.git
```
Or:
```shell
$ git clone git@github.com:milvus-io/milvus-sdk-cpp.git
```

### Install the dependencies

```shell
$ cd milvus-sdk-cpp
$ bash scripts/install_deps.sh
```

This script could help you set a developing environment from a minimal installation.

## Building from source

You could build the debug versioned SDK with `make` in the source directory, or `make all-release` to build the release version.

And you could also create a dedicated CMake build directory, then use CMake to build it from the source by yourself

```shell
$ cd milvus-sdk-cpp
$ mkdir cmake_build
$ cd cmake_build
$ cmake ..
$ make
```

## Building with external gRPC

By default, `make` command downloads gRPC source code from github and compile it under the CMake build directory.

You can use a pre-built gRPC lib build by youself. milvus-sdk-cpp 2.4 is using gRPC v1.59.0, make sure your gRPC version is compatible.

### Download gRPC source code
```shell
$ git clone https://github.com/grpc/grpc.git
$ git checkout v1.59.0
$ git submodule update --init
```
Or:
```shell
$ git clone git@github.com:grpc/grpc.git
$ git checkout v1.59.0
$ git submodule update --init
```

### Build gRPC dynamic lib from source code and install it
```shell
$ cd grpc
$ mkdir cmake_build
$ cd cmake_build
$ cmake -DCMAKE_INSTALL_PREFIX=/path/to/pre-installed/grpc  -DBUILD_SHARED_LIBS=ON ..
$ make
$ make install
```
Make sure the `BUILD_SHARED_LIBS` is `ON` since milvus-sdk-cpp dynamiclly links to gRPC.

### Use `GRPC_PATH` to specify the external gRPC and build milvus-sdk-cpp
```shell
$ cd milvus-sdk-cpp
$ mkdir cmake_build
$ cd cmake_build
$ cmake ..
$ make GRPC_PATH=/path/to/pre-installed/grpc
```
With `GRPC_PATH`, milvus-sdk-cpp will skip gRPC downloading and compile/link with the external gRPC.

## Code style for Milvus C++ SDK
Milvus C++ SDK project using the similar clang-format and clang-tidy rules
from [milvus-io/milvus](https://github.com/milvus-io/milvus)

We have defined some naming rules in clang-tidy rules.

Using `make lint` under the source directory helps you to check your local modification
if compliance with cpplint/clang-format/clang-tidy.

You could also execute the command `make clang-format` under the CMake build directory
to automatic format all c++ source code


## Run tests, and add testing code
Milvus C++ SDK using googletest as a test framework. You could run `make test` to run unit testing and integration testing.

If you have an pre-installed gRPC, use `GRPC_PATH` to specify the path:
```shell
$ make test GRPC_PATH=/path/to/pre-installed/grpc
```

If you add some new code, you'd better add related testing code together.
We have below test scopes:
- Test code under `test/ut`: the code could run without any Milvus server, which we called unit testing. 
- Test code under `test/it`: the code needs to run with a mocked server, which we called integration testing.
- Test code under `test/st`: the code needs to run with a real Milvus server, which we called that acceptance testing.

The test cases are built as executable binaries under the path `cmake_build/test`:
```shell
$ ./cmake_build/test/testing-it
$ ./cmake_build/test/testing-ut
$ ./cmake_build/test/testing-st
```

### Run acceptance/system tests with real Milvus server
The acceptance/system tests are not included by default. You cloud using the below commands to run them:
- `make st` user the top source directory
- `make system-test` under the CMake build directory

The acceptance/system tests will startup container by docker, and using jq to capture the output from docker inspect,
so you need to install docker and jq tools for running them.


## Try the examples
Once the `make test` is done, you will see some executable examples under the path `./cmake_build/examples`.
- `./cmake_build/examples/sdk_array`: example to show the usage of Array field.
- `./cmake_build/examples/sdk_db`: example to show the usage of databases.
- `./cmake_build/examples/sdk_dml`: example to show the usage of dml interfaces.
- `./cmake_build/examples/sdk_general`: a general example to show the basic usage.
- `./cmake_build/examples/sdk_hybrid_search`: example to show the usage of hybrid search interface.
- `./cmake_build/examples/sdk_iterator_query`: example to show the usage of query iterator.
- `./cmake_build/examples/sdk_iterator_search`: example to show the usage of search iterator.
- `./cmake_build/examples/sdk_json`: example to show the usage of JSON field.
- `./cmake_build/examples/sdk_partition_key`: example to show the usage of partition key.
- `./cmake_build/examples/sdk_rbac`: example to show the usage of RBAC.
- `./cmake_build/examples/sdk_vector_binary`: example to show the usage of BinaryVector field.
- `./cmake_build/examples/sdk_vector_fp16`: example to show the usage of Float16Vector/BFloat16Vector field.
- `./cmake_build/examples/sdk_vector_sparse`: example to show the usage of SparseVector field.


## Run code coverage
Milvus C++ SDK using **lcov** tool to generate code coverage report. You could run `make coverage`, this command will:
- run all unittest cases
- generate code coverage report by lcov tool

After the command, a folder named "code_coverage" will be created under the project.
You could open the **code_coverage/index.html** by a web browser to review the code coverage report.

## Generate documentation
Milvus C++ SDK using **doxygen** tool to generate documentation. Run `make documentation` to generate documentation.
Typically, we only publish documentation before releasing a new sdk version.
Since the **doxygen** is not included in the `install_deps.sh`, you need to manually install it if you want to generate the documentation by yourself.

# Build C++ SDK with your macOS

## Setup development environment

The setup steps and development environment for macOS are similar to Linux.
You could use `install_deps.sh` to install dependencies and use the same `make` commands for build, lint, test, and coverage.

### Prerequests
Before you run `install_deps.sh` to install dependencies, you should make sure:
- Already installed (Homebrew)[https://brew.sh/]
- Install Command line tools for Xcode, by command: `xcode-select --instal`
