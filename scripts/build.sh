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
BUILD_UNIT_TEST="OFF"
MAKE_CLEAN="OFF"
RUN_CPPLINT="OFF"

while getopts "p:d:t:f:ulrcgjhxzme" arg; do
  case $arg in
  t)
    BUILD_TYPE=$OPTARG # BUILD_TYPE
    ;;
  l)
    RUN_CPPLINT="ON"
    BUILD_UNIT_TEST="ON"  # lint requires build with ut
    ;;
  r)
    if [[ -d ${BUILD_OUTPUT_DIR} ]]; then
      rm ./${BUILD_OUTPUT_DIR} -r
      MAKE_CLEAN="ON"
    fi
    ;;
  u)
    UNIT_TEST="ON"
    BUILD_UNIT_TEST="ON"
    ;;
  h) # help
    echo "

parameter:
-t: build type(default: Debug)
-l: run cpplint, clang-format and clang-tidy(default: OFF)
-u: build with unit testing(default: OFF)
-h: help

usage:
./build.sh -t \${BUILD_TYPE} [-l] [-r] [-h]
                "
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

CMAKE_CMD="cmake \
-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
-DBUILD_UNIT_TEST=${BUILD_UNIT_TEST}
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
  make -j 4 || exit 1
  make check-clang-tidy
  if [ $? -ne 0 ]; then
    echo "ERROR! clang-tidy check failed"
    exit 1
  fi
  echo "clang-tidy check passed!"
else
  # compile and build
  make -j 4  || exit 1
fi

if [[ "${UNIT_TEST}" == "ON" ]]; then
  make -j 4  || exit 1
  make CTEST_OUTPUT_ON_FAILURE=1 test || exit 1
fi
