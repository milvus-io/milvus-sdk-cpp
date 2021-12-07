# Development Guide Milvus C++ SDK
This document will help to set up your development environment and run tests for Milvus C++ SDK. If you encounter a problem, please file an [issue](https://github.com/milvus-io/milvus-sdk-cpp/issues/new).

# Build C++ SDK with your Linux

## Setup development environment
Currently we tested below platform and compilers for developing Milvus C++ SDK.

| Platform | Version       | Compiler Tested | Support                       |
|----------|---------------|-----------------|-------------------------------|
| Linux    | Ubuntu 18.04  | GCC 7.0.0       | Full (Compile, Lint, Testing) |
| Linux    | CentOS        | GCC 4.8.5       | Compile, Testing              |

### Clone the code

```shell
$ git clone git@github.com:milvus-io/milvus-sdk-cpp.git
```

### Install the dependencies

```shell
$ cd milvus-sdk-cpp
$ bash scrips/install_deps.sh
```

This scripts could help you to setup a development environment even from a minimal installation.

## Build SDK
You could simply build the debug version sdk with `make` in the source directory.

Or `make all-release` to build the release version.

And you could also create a cmake build directory, and using cmake to build it from source by yourself

```shell
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## Code style for Milvus C++ SDK
Milvus C++ SDK project using the similar clang-format and clang-tidy rules
from [milvus-io/milvus](https://github.com/milvus-io/milvus)

We also have some naming rule, which already defined in clang-tidy rules.

Using `make lint` under source directory will helps you to check you local modification
if compliance with cpplint/clang-format/clang-tidy.

You could also execute command `make clang-format` under the cmake build directory
to automaic format all c++ source code


## Run tests, and add testing code
Milvus C++ SDK using googletest as test framework. You could run `make test` to run all tests.

If you add some new code, you'd better add related testing code together.
We have below test scopes:
- Test code under `test/ut` the code could be run without any Milvus server. 
- Test code under `test/it` the code need to be run with a mocked server,
  generally we called integration testing.
- Test code under `test/st` the code need to be run with a real Milvus server,
  generally we called that acceptance testing.
