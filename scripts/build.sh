#!/usr/bin/env bash

# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

BUILD_OUTPUT_DIR="cmake_build"
BUILD_TYPE="Debug"
UNIT_TEST="OFF"
SYS_TEST="OFF"
BUILD_TEST="OFF"
MAKE_CLEAN="OFF"
RUN_FORMAT="ON"
RUN_CPPLINT="OFF"
BUILD_COVERAGE="OFF"
MILVUS_SDK_VERSION=${MILVUS_SDK_VERSION:-}
DO_INSTALL="OFF"
CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX:-/usr/local}
BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:-ON}
BUILD_FROM_CONAN="ON"
CPPSTD=${CPPSTD:-14}

JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 3)}"
if [ ${JOBS} -lt 3 ] ; then
    JOBS=3
fi
if [ ${JOBS} -gt 10 ] ; then
    JOBS=10
fi

while getopts "t:v:ulrcsphizf" arg; do
  case $arg in
  t)
    BUILD_TYPE=$OPTARG # BUILD_TYPE
    ;;
  v)
    MILVUS_SDK_VERSION=$OPTARG
    ;;
  l)
    RUN_CPPLINT="ON"
    BUILD_TEST="ON"  # lint requires build with ut
    ;;
  r)
    if [[ -d ${BUILD_OUTPUT_DIR} ]]; then
      rm ./${BUILD_OUTPUT_DIR} -rf
      MAKE_CLEAN="ON"
    fi
    ;;
  u)
    UNIT_TEST="ON"
    BUILD_TEST="ON"
    ;;
  c)
    BUILD_COVERAGE="ON"
    ;;
  s)
    SYS_TEST="ON"
    BUILD_TEST="ON"
    ;;
  p)
    BUILD_TYPE=RelWithDebInfo
    RUN_CPPLINT="OFF"
    SYS_TEST="OFF"
    UNIT_TEST="OFF"
    BUILD_TEST="OFF"
    MAKE_CLEAN="ON"
    ;;
  i)
    DO_INSTALL="ON"
    ;;
  z)
    BUILD_FROM_CONAN="OFF"
    ;;
  f)
    # Disable in-place clang-format of source files. Used by reproducible
    # builds (e.g. the package / release target) that must not mutate the
    # working tree.
    RUN_FORMAT="OFF"
    ;;
  h) # help
    echo "

parameter:
-t: build type(default: Debug)
-v: sdk version override(default: latest git tag from CMake)
-l: run cpplint, clang-format and clang-tidy(default: OFF)
-u: build with unit testing(default: OFF)
-r: clean before build
-s: build with system testing(default: OFF)
-c: build with coverage
-p: build with production(-t RelWithDebInfo -r)
-i: do install
-z: build without conan-managed dependencies
-f: skip in-place clang-format of source files (for reproducible builds)
-h: help

usage:
./build.sh -t \${BUILD_TYPE} -v \${MILVUS_SDK_VERSION} [-l] [-r] [-r] [-s] [-p] [-h]"
    exit 0
    ;;
  ?)
    echo "ERROR! unknown argument"
    exit 1
    ;;
  esac
done

if [[ ! -d ${BUILD_OUTPUT_DIR} ]]; then
  mkdir ${BUILD_OUTPUT_DIR}
fi

cd ${BUILD_OUTPUT_DIR}

# remove make cache since build.sh -l use default variables
# force update the variables each time
make rebuild_cache >/dev/null 2>&1

if [[ "${BUILD_FROM_CONAN}" == "ON" ]]; then
  echo "Use Conan-managed dependencies"
  # Conan 2 integration (Conan-only dependency management).
  # Dependencies must come from Conan; external gRPC is not supported.
  # Users can override the Conan executable via CONAN.
  CONAN=${CONAN:-conan}
  CONAN_LIBCXX_SETTINGS=()
  if [[ -n "${CONAN_LIBCXX:-}" ]]; then
    CONAN_LIBCXX_SETTINGS+=("-s" "compiler.libcxx=${CONAN_LIBCXX}")
  fi
  # Host profile uses CPPSTD (default 14) — applies to everything linked into
  # libmilvus_sdk.so, so the ABI matches the SDK's own compilation.
  #
  # Build profile uses a newer C++ standard for build-time tools only
  # (protoc, grpc_cpp_plugin, cmake, and their transitive deps). Recent
  # versions of these tools pull in a newer abseil that ConanCenter only
  # publishes with a higher cppstd; requesting a lower one triggers
  # "compatible packages" fallback and protobuf's cppstd-match validation
  # then rejects the combination. Setting the build profile to the cppstd
  # ConanCenter actually publishes makes Conan hit the cache directly.
  #
  # This is purely a build-time concern — nothing from the build profile
  # ends up in libmilvus_sdk.so, so it doesn't affect the SDK's ABI or
  # what C++ standard consumers need.
  BUILD_CPPSTD=17

  # Conan install generates a toolchain and CMake dependency config files.
  ${CONAN} --version || exit 1

  if [[ "${BUILD_TEST}" == "ON" ]]; then
    CONAN_WITH_TESTS=True
  else
    CONAN_WITH_TESTS=False
  fi

  # Build folder layout follows Conan 2 CMakeToolchain defaults:
  #   cmake_build/build/<BuildType>/generators/conan_toolchain.cmake
  # Workaround: CMake 4.x removed compat with cmake_minimum_required < 3.5,
  # which breaks older Conan recipes (e.g. c-ares/1.19.1).
  export CMAKE_POLICY_VERSION_MINIMUM=3.5

  ${CONAN} install .. \
    -of . \
    -s build_type=${BUILD_TYPE} \
    -s compiler.cppstd=${CPPSTD} \
    "${CONAN_LIBCXX_SETTINGS[@]}" \
    -s:b build_type=${BUILD_TYPE} \
    -s:b compiler.cppstd=${BUILD_CPPSTD} \
    -o "&:with_tests=${CONAN_WITH_TESTS}" \
    -c tools.build:jobs=${JOBS} \
    --build=missing || exit 1

  TOOLCHAIN_FILE="${PWD}/build/${BUILD_TYPE}/generators/conan_toolchain.cmake"
  if [ ! -f "${TOOLCHAIN_FILE}" ]; then
    echo "ERROR! Conan toolchain not found at ${TOOLCHAIN_FILE}"
    echo "Make sure Conan 2 is installed and 'conan install' finished successfully."
    exit 1
  fi

else
  if [ -z "$GRPC_PATH" ]; then
    echo "Use internal gRPC package"
  else
    echo "External gRPC path: ${GRPC_PATH}"
    # set GRPC_PATH into path environment since the build process will call protoc to compile milvus porot files
    # and the protoc executable requires some libraries under the gRPC install path
    unameOut="$(uname -s)"
    case "${unameOut}" in
        Linux*)
          export LD_LIBRARY_PATH="${GRPC_PATH}/lib:${LD_LIBRARY_PATH}"
          ;;
        Darwin*)
          export DYLD_LIBRARY_PATH="${GRPC_PATH}/lib:${DYLD_LIBRARY_PATH}"
          ;;
        MINGW*)
          export LD_LIBRARY_PATH="${GRPC_PATH}/lib;${LD_LIBRARY_PATH}"
          ;;
        *)
          echo "gRPC path is not passed into environment path"
    esac
  fi
fi

CMAKE_VERSION_ARG=""
if [[ -n "${MILVUS_SDK_VERSION}" ]]; then
  CMAKE_VERSION_ARG="-DMILVUS_SDK_VERSION=${MILVUS_SDK_VERSION}"
fi

CMAKE_BUILD_EXAMPLES=${CMAKE_BUILD_EXAMPLES:-ON}
case "${CMAKE_BUILD_EXAMPLES}" in
  ON|OFF)
    ;;
  *)
    echo "ERROR! CMAKE_BUILD_EXAMPLES must be ON or OFF"
    exit 1
    ;;
esac

CMAKE_CMD="cmake \
-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
-DCMAKE_CXX_STANDARD=${CPPSTD} \
-DMILVUS_BUILD_TEST=${BUILD_TEST} \
-DMILVUS_BUILD_COVERAGE=${BUILD_COVERAGE} \
-DMILVUS_BUILD_EXAMPLES=${CMAKE_BUILD_EXAMPLES} \
${CMAKE_VERSION_ARG} \
-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} \
-DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} \
-DGRPC_PATH=${GRPC_PATH} \
-DBUILD_FROM_CONAN=${BUILD_FROM_CONAN} \
../"
echo ${CMAKE_CMD}
${CMAKE_CMD} || exit 1

# if centos 7, enable devtoolset-7
if [ -f /opt/rh/devtoolset-7/enable ] ; then
  source /opt/rh/devtoolset-7/enable
fi

if [[ ${MAKE_CLEAN} == "ON" ]]; then
  make clean
fi

if [[ ${RUN_FORMAT} == "ON" ]]; then
  find ../src/include/milvus ../src/impl ../test ../examples -iname '*.h' -o -iname '*.cpp' -o -iname '*.hpp' | xargs make clang-format -i
fi

if [[ ${RUN_CPPLINT} == "ON" ]]; then
  # cpplint check
  make lint
  if [ $? -ne 0 ]; then
    echo "ERROR! cpplint check failed"
    exit 1
  fi
  echo "cpplint check passed!"

  # clang-format check
  make check-clang-format
  if [ $? -ne 0 ]; then
    echo "ERROR! clang-format check failed"
    exit 1
  fi
  echo "clang-format check passed!"

  # clang-tidy check
  make -j ${JOBS} || exit 1
  make check-clang-tidy
  if [ $? -ne 0 ]; then
    echo "ERROR! clang-tidy check failed"
    exit 1
  fi
  echo "clang-tidy check passed!"
else
  # compile and build
  make -j ${JOBS}  || exit 1
fi

if [[ "${UNIT_TEST}" == "ON" ]]; then
  make -j ${JOBS}  || exit 1
  # Suppress gRPC verbose logs during tests
  GRPC_VERBOSITY=ERROR ./test/testing-ut || exit 1
  GRPC_VERBOSITY=ERROR ./test/testing-it || exit 1
fi

if [[ "${SYS_TEST}" == "ON" ]]; then
  make -j ${JOBS}  || exit 1
  # Suppress gRPC verbose logs during tests
  GRPC_VERBOSITY=ERROR ./test/testing-st || exit 1
fi

if [[ "${DO_INSTALL}" == "ON" ]]; then
  make install || exit 1
fi
