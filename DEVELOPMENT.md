# Milvus C++ SDK Development Guide

This document explains how to set up a development environment and run tests for the Milvus C++ SDK.
Please file an [issue](https://github.com/milvus-io/milvus-sdk-cpp/issues/new) if you have any questions.

## Supported platforms

- Linux
- macOS
- Windows

## Tested development environments

The following environments are currently exercised by CI. Compiler patch versions may change when
runner images or distribution packages are updated.

| Platform | Environment | Compiler | CI coverage |
|---|---|---|---|
| Linux | Ubuntu 20.04, AMD64 | GCC 9.4.0 | Lint, unit and mocked tests, package verification |
| Linux | Ubuntu 22.04, AMD64 | GCC 11.4 | Unit, mocked, and system tests; coverage |
| Linux | Fedora 39, AMD64 | GCC 13.3.1 | Unit and mocked tests, package verification |
| macOS | macOS 15, ARM64 | Apple Clang 17 | Unit and mocked tests, package verification |
| Windows | Windows Server 2022, AMD64 | MSVC 2022 | Build, unit tests, install and package verification |

Windows CI does not run mocked integration tests because the in-process mocked gRPC server is
unstable across the Windows executable/DLL boundary.

## Set up a development environment

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

This script installs the development dependencies on supported platforms.

## Building from source

You can build the debug versioned SDK with `make` in the source directory, or `make all-release` to build the release version.

By default, `make` uses [Conan 2](https://conan.io/) to manage dependencies (gRPC, protobuf, abseil, etc.). The top-level `Makefile` delegates to `scripts/build.sh`, which handles Conan integration and CMake configuration automatically.

Common top-level make targets:

```shell
$ make                    # debug build, same as make all-debug
$ make all-release        # release build
$ make test               # debug build with unit tests and mocked integration tests enabled
$ make test-release       # release build with unit tests and mocked integration tests enabled
$ make lint               # build with unit tests and run lint checks
$ make package            # release build and create DEB/RPM packages under cmake_build/Pack
$ make clean              # remove build directories
```

You can pass environment variables through `make` to control `scripts/build.sh` and CMake:

```shell
$ MILVUS_SDK_VERSION=v3.0.2 make all-release
$ CMAKE_INSTALL_PREFIX=/opt/milvus make install
$ BUILD_SHARED_LIBS=OFF make all-release
$ CPPSTD=17 make test
$ UNITY=ON LINE=ON JOBS=4 make test
```
`scripts/build.sh` also supports command-line flags. Most users should prefer the make targets above, but the script can be invoked directly for more control:

```shell
$ bash scripts/build.sh -t Debug            # debug build
$ bash scripts/build.sh -t Release          # release build
$ bash scripts/build.sh -u                  # build and run unit tests
$ bash scripts/build.sh -l                  # run cpplint, clang-format, and clang-tidy checks
$ bash scripts/build.sh -r                  # clean before build
$ bash scripts/build.sh -z                  # build without Conan-managed dependencies
$ bash scripts/build.sh -f -t Release       # release build without in-place clang-format
$ bash scripts/build.sh -v v3.0.2 -t Release
```
`MILVUS_SDK_VERSION` is embedded into the SDK version string generated from `src/impl/version.h.in`. If you do not pass `MILVUS_SDK_VERSION` or `scripts/build.sh -v`, CMake derives the version from the latest reachable git tag. For release builds, pass the intended version explicitly or build from the release tag.

## Speed up local builds

Two make-level switches reduce cold-build wall time without changing runtime behavior. They
apply to targets that build the SDK through `scripts/build.sh` (for example `make`,
`make test`, `make coverage`, and `make package`) and have no effect on targets such as
`doc`, `clean`, `tutorials`, or the `run` wrappers:

- `UNITY=ON` maps to CMake `MILVUS_ENABLE_UNITY=ON` and enables unity builds for the SDK object
  library. Many request/response translation units are small glue files, and unity batches pay
  their shared header closure once per batch instead of once per file. Generated protobuf
  sources are always excluded from the batches. Trade-off: editing one `.cpp` recompiles its
  whole batch, so this mode mainly helps cold and CI builds rather than incremental edit cycles.
  Keep it off for coverage runs: unity batches can attribute lines to the generated batch files
  and distort the lcov report (`build.sh` prints a warning for this combination).
- `LINE=ON` maps to CMake `MILVUS_DEBUG_INFO=line-tables`, which keeps stack traces and
  `file:line` debug information but skips variable-level DWARF (`-gline-tables-only` on Clang,
  `-g1` on GCC). It applies to Debug and RelWithDebInfo builds. Switch back to `LINE=OFF` when
  you need to inspect variables in a debugger. It is also safe for coverage builds: gcov
  coverage data comes from the `.gcno`/`.gcda` notes files, not DWARF (validated by identical
  lcov output between full and line-tables debug info).

Both default to `OFF`, so regular builds are unchanged. Measured cold-build effect for
`UNITY=ON LINE=ON` versus defaults: about 2.6x faster with roughly half the disk usage. Use
`bash scripts/build.sh -r` (or remove `cmake_build`) when switching modes, because the flags are
captured at CMake configure time. `MILVUS_UNITY_BATCH_SIZE` (default 8) tunes the unity batch
size for finer control.

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
Examples are not built for coverage runs (they are filtered out of the report anyway);
set `CMAKE_BUILD_EXAMPLES=ON` explicitly if you need them. Set `COVERAGE_HTML=OFF` to
skip the HTML report and keep only `code_coverage/lcov_output.info`. The lcov capture
runs across `COVERAGE_JOBS` workers (default: all CPUs) and produces output identical
to the sequential run.

## Generate documentation
Milvus C++ SDK uses **doxygen** tool to generate documentation. Run `make doc` to generate documentation.
After the command, open **doc/html/index.html** in a web browser to view the documentation.
Typically, we only publish documentation before releasing a new sdk version.
Since the **doxygen** is not included in the `install_deps.sh`, you need to manually install it if you want to generate the documentation by yourself.

## Build the C++ SDK on macOS

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

## Build the C++ SDK on Windows

## Prerequisites
- Visual Studio 2022 with C++ workload
- [CMake](https://cmake.org/) 3.16+
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
