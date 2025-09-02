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
RUN_CPPLINT="OFF"
BUILD_COVERAGE="OFF"
MILVUS_SDK_VERSION=${MILVUS_SDK_VERSION:-2.4.0}
DO_INSTALL="OFF"
CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX:-/usr/local}
BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:-ON}

JOBS="$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 3)"
if [ ${JOBS} -lt 3 ] ; then
    JOBS=3
fi

while getopts "t:v:ulrcsphi" arg; do
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
  h) # help
    echo "

parameter:
-t: build type(default: Debug)
-l: run cpplint, clang-format and clang-tidy(default: OFF)
-u: build with unit testing(default: OFF)
-r: clean before build
-s: build with system testing(default: OFF)
-c: build with coverage
-p: build with production(-t RelWithDebInfo -r)
-i: do install
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

# if centos 7, enable devtoolset-7
if [ -f /opt/rh/devtoolset-7/enable ] ; then
  source /opt/rh/devtoolset-7/enable
fi

# if the external gRPC is specified, the testing binaries require the LD_LIBRARY_PATH to include gRPC lib path
# and the protoc exe also needs the path to dynamic link dependencies
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${GRPC_PATH}/lib

CMAKE_CMD="cmake \
-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
-DMILVUS_BUILD_TEST=${BUILD_TEST} \
-DMILVUS_BUILD_COVERAGE=${BUILD_COVERAGE} \
-DMILVUS_SDK_VERSION=${MILVUS_SDK_VERSION} \
-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX} \
-DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} \
-DGRPC_PATH=${GRPC_PATH} \
-DLD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
../"
echo ${CMAKE_CMD}
${CMAKE_CMD}

if [[ ${MAKE_CLEAN} == "ON" ]]; then
  make clean
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
  ./test/testing-st || exit 1
fi

if [[ "${DO_INSTALL}" == "ON" ]]; then
  make install || exit 1
fi
