# Development Guide Milvus C++ SDK
This document helps you set up your development environment and run tests for Milvus C++ SDK.
Please file an [issue](https://github.com/milvus-io/milvus-sdk-cpp/issues/new) if you have any questions.

# Supported Platforms
- Linux
- macOS

# Build C++ SDK with your Linux

## Setup development environment
Currently, we tested the below platform and compilers for developing Milvus C++ SDK.

| Platform | Version      | Compiler Tested | Support                       |
| -------- | ------------ | --------------- | ----------------------------- |
| Linux    | Ubuntu 18.04 | GCC 7.0.0       | Full (Compile, Lint, Testing) |
| Linux    | Ubuntu 20.04 | GCC 9.3.0       | Full (Compile, Lint, Testing) |
| Linux    | Fedora 34/35 | GCC 11.2        | Compile, Testing              |
| Linux    | CentOS 7     | GCC 4.8.5       | Compile, Testing              |

### Clone the code

```shell
$ git clone git@github.com:milvus-io/milvus-sdk-cpp.git
```

### Install the dependencies

```shell
$ cd milvus-sdk-cpp
$ bash scrips/install_deps.sh
```

This script could help you set a developing environment from a minimal installation.

## Build SDK
You could build the debug versioned SDK with `make` in the source directory.

Or `make all-release` to build the release version.

And you could also create a dedicated CMake build directory, then use CMake to build it from the source by yourself

```shell
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## Code style for Milvus C++ SDK
Milvus C++ SDK project using the similar clang-format and clang-tidy rules
from [milvus-io/milvus](https://github.com/milvus-io/milvus)

We have defined some naming rules in clang-tidy rules.

Using `make lint` under the source directory helps you to check your local modification
if compliance with cpplint/clang-format/clang-tidy.

You could also execute the command `make clang-format` under the CMake build directory
to automatic format all c++ source code


## Run tests, and add testing code
Milvus C++ SDK using googletest as a test framework. You could run `make test` to run all tests.

If you add some new code, you'd better add related testing code together.
We have below test scopes:
- Test code under `test/ut`: the code could run without any Milvus server. 
- Test code under `test/it`: the code needs to run with a mocked server, which we called integration testing.
- Test code under `test/st`: the code needs to run with a real Milvus server, which we called that acceptance testing.

### Run acceptance/system tests with real Milvus server
The acceptance/system tests are not included by default. You cloud using the below commands to run them:
- `make st` user the top source directory
- `make system-test` under the CMake build directory

The acceptance/system tests will startup container by docker, and using jq to capture the output from docker inspect,
so you need to install docker and jq tools for running them.

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
