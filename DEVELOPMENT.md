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
| Linux    | Ubuntu 20.04 | GCC 9.3.0            | Full (Compile, Lint, Testing) |
| Linux    | Ubuntu 22.04 | GCC 11.4             | Full (Compile, Lint, Testing) |
| Linux    | Fedora 38/39 | GCC 11.2+            | Compile, Testing              |
| macOS    | macOS 14     | Apple Clang           | Compile, Testing              |
| Windows  | Windows 2022 | MSVC 2022            | Compile, Testing              |

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

By default, `make` uses [Conan 2](https://conan.io/) to manage dependencies (gRPC, protobuf, abseil, etc.). The `scripts/build.sh` handles Conan integration automatically.

## Building without Conan

If you prefer not to use Conan, the following targets will download and compile gRPC from source:
```shell
$ make build-no-conan-debug    # debug build
$ make build-no-conan-release  # release build
$ make test-no-conan           # build and run unit tests
```

## Code style for Milvus C++ SDK
Milvus C++ SDK project using the similar clang-format and clang-tidy rules
from [milvus-io/milvus](https://github.com/milvus-io/milvus)

We have defined some naming rules in clang-tidy rules.

Make sure you have installed clang-format and clang-tidy:
```
sudo apt install clang-format clang-tidy
```

Using `make lint` under the source directory helps you to check your local modification
if compliance with cpplint/clang-format/clang-tidy.

You could also execute the command `make clang-format` under the CMake build directory
to automatic format all c++ source code


## Run tests, and add testing code
Milvus C++ SDK using googletest as a test framework. You could run `make test` to run unit testing and integration testing.

If you have a pre-installed gRPC, use `GRPC_PATH` to specify the path:
```shell
$ make test GRPC_PATH=/path/to/pre-installed/grpc
```

If you add some new code, you'd better add related testing code together.
We have below test scopes:
- Test code under `test/ut`: unit testing, tests run without any server.
- Test code under `test/it`: mock testing, tests run with a mocked gRPC server.
- Test code under `test/st`: integration testing, tests run with a real Milvus server via Docker.

The test cases are built as executable binaries under the path `cmake_build/test`:
```shell
$ ./cmake_build/test/testing-it
$ ./cmake_build/test/testing-ut
$ ./cmake_build/test/testing-st
```

### Run acceptance/system tests with real Milvus server
The acceptance/system tests are not included by default. You could use the below commands to run them:
- `make st` under the top source directory
- `make system-test` under the CMake build directory

The acceptance/system tests will start a Milvus container via Docker automatically.
You need Docker installed and the Python Docker SDK (`pip install docker`) for running them.


## Try the examples
Once the `make test` is done, you will see some executable examples under the path `./cmake_build/examples`.
See [Examples Guide](examples/README.md) for details.


## Run code coverage
Milvus C++ SDK using **lcov** tool to generate code coverage report. You could run `make coverage`, this command will:
- run all unittest cases
- generate code coverage report by lcov tool

After the command, a folder named "code_coverage" will be created under the project.
You could open the **code_coverage/index.html** by a web browser to review the code coverage report.

## Generate documentation
Milvus C++ SDK uses **doxygen** tool to generate documentation. Run `make doc` to generate documentation.
After the command, open **doc/html/index.html** in a web browser to view the documentation.
Typically, we only publish documentation before releasing a new sdk version.
Since the **doxygen** is not included in the `install_deps.sh`, you need to manually install it if you want to generate the documentation by yourself.

# Build C++ SDK with your macOS

## Prerequisites
- [Homebrew](https://brew.sh/)
- Command line tools for Xcode: `xcode-select --install`
- Python 3 with pip

## Setup development environment

Install dependencies using the provided script:
```shell
$ mkdir ~/.venv
$ python3 -m venv ~/.venv
$ source ~/.venv/bin/activate
$ bash scripts/install_deps.sh
```

Note: Starting from macOS 14 (Sonoma), Apple prevents pip from modifying system directories.
A Python virtual environment is required for installing build tools (cmake, clang-format, clang-tidy).

The script installs the following via Homebrew: `wget`, `lcov`, `llvm`, `openssl@3`, `ccache`.

## Building and testing

You can build with Conan (same as Linux):
```shell
$ make          # build with Conan-managed dependencies
$ make test     # build and run unit tests + mock tests
```

Or build without Conan:
```shell
$ make test-no-conan           # build and run tests without Conan
$ make build-no-conan-debug    # debug build without Conan
$ make build-no-conan-release  # release build without Conan
```

# Build C++ SDK with your Windows

## Prerequisites
- Visual Studio 2022 with C++ workload
- [CMake](https://cmake.org/) 3.14+
- [Ninja](https://ninja-build.org/) build system
- [ccache](https://ccache.dev/) (optional, for faster rebuilds)

You can install CMake, Ninja, and ccache via [Chocolatey](https://chocolatey.org/):
```cmd
choco install cmake ninja ccache
```

## Building and testing

Open a **Developer Command Prompt for VS 2022** (or **x64 Native Tools Command Prompt**), then:
```cmd
cmake -S . -B build -DMILVUS_BUILD_TEST=YES -G Ninja
cmake --build build
```

Run tests:
```cmd
build\test\testing-ut
build\test\testing-it
```

Note: The Windows build does not use Conan. CMake downloads and compiles gRPC from source automatically. Conan-based build and system tests (`testing-st`) are not supported on Windows.
