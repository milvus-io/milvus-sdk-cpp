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
MILVUS_SDK_VERSION=${MILVUS_SDK_VERSION:-v2.6.1}
DO_INSTALL="OFF"
CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX:-/usr/local}
BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:-ON}
BUILD_FROM_CONAN="ON"
MILVUS_WITH_TESTCONTAINERS=${MILVUS_WITH_TESTCONTAINERS:-OFF}


JOBS="$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 3)"
if [ ${JOBS} -lt 3 ] ; then
    JOBS=3
fi
if [ ${JOBS} -gt 10 ] ; then
    JOBS=10
fi

while getopts "t:v:ulrcsphizT" arg; do
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
    MILVUS_WITH_TESTCONTAINERS="ON"
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
  T)
    MILVUS_WITH_TESTCONTAINERS="ON"
    ;;
  h) # help
    echo "

parameter:
-t: build type(default: Debug)
-l: run cpplint, clang-format and clang-tidy(default: OFF)
-u: build with unit testing(default: OFF)
-r: clean before build
-s: build with system testing(default: OFF)
-T: enable Testcontainers for system tests(default: OFF)
-c: build with coverage
-p: build with production(-t RelWithDebInfo -r)
-i: do install
-z: disable conan build
-h: help

usage:
./build.sh -t \${BUILD_TYPE} -v \${MILVUS_SDK_VERSION} [-l] [-r] [-r] [-s] [-T] [-p] [-i] [-z] [-h]"
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
  CPPSTD=${CPPSTD:-14}

  # Conan install generates a toolchain and CMake dependency config files.
  # Use both host and build profiles with the same C++ standard to avoid mismatches.
  ${CONAN} --version || exit 1

  if [[ "${BUILD_TEST}" == "ON" ]]; then
    CONAN_WITH_TESTS=True
  else
    CONAN_WITH_TESTS=False
  fi

  # Build folder layout follows Conan 2 CMakeToolchain defaults:
  #   cmake_build/build/<BuildType>/generators/conan_toolchain.cmake
  ${CONAN} install .. \
    -of . \
    -s build_type=${BUILD_TYPE} \
    -s compiler.cppstd=${CPPSTD} \
    -s:b build_type=${BUILD_TYPE} \
    -s:b compiler.cppstd=${CPPSTD} \
    -o "&:with_tests=${CONAN_WITH_TESTS}" \
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

CMAKE_CMD="cmake \
-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE} \
-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
-DMILVUS_BUILD_TEST=${BUILD_TEST} \
-DMILVUS_BUILD_COVERAGE=${BUILD_COVERAGE} \
-DMILVUS_WITH_TESTCONTAINERS=${MILVUS_WITH_TESTCONTAINERS} \
-DMILVUS_SDK_VERSION=${MILVUS_SDK_VERSION} \
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
  ./test/testing-ut || exit 1
  ./test/testing-it || exit 1
fi

if [[ "${SYS_TEST}" == "ON" ]]; then
  make -j ${JOBS}  || exit 1
  if [[ "${MILVUS_WITH_TESTCONTAINERS}" == "ON" ]]; then
    TESTCONTAINERS_LIB_DIR="${PWD}/_deps/testcontainers_native-build/testcontainers-c"
    if [[ -d "${TESTCONTAINERS_LIB_DIR}" ]]; then
      export LD_LIBRARY_PATH="${TESTCONTAINERS_LIB_DIR}:${LD_LIBRARY_PATH}"
      export DYLD_LIBRARY_PATH="${TESTCONTAINERS_LIB_DIR}:${DYLD_LIBRARY_PATH}"
    fi
  fi
  pushd test >/dev/null || exit 1
  ./testing-st || exit 1
  popd >/dev/null || exit 1
fi

if [[ "${DO_INSTALL}" == "ON" ]]; then
  make install || exit 1
fi
