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

cmake_minimum_required(VERSION 3.12)
project(milvus_sdk LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

# enable ccache if possible
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
    message(STATUS "using ccache")
endif(CCACHE_FOUND)

include(FindClangTools)
include(GoogleTest)
include(CTest)

include(DefineOptions)

# options

set_option_category("Build")

define_option(MILVUS_BUILD_TEST "Build with testing" OFF)
define_option(MILVUS_BUILD_COVERAGE "Build with coverage" OFF)
define_option_string(MILVUS_SDK_VERSION
    "Version for sdk"
    "2.0.0")
define_option_string(MILVUS_SDK_RELEASE
    "Release number for sdk"
    "1")

set_option_category("Thirdparty")
# use thirdparty from:
#  auto: Find -> Fetch_Content_Then_Find
#  package: Find
#  module: Fetch_Content_Then_Find
define_option_string(MILVUS_WITH_GRPC           "Using gRPC from"       "module" "package" "module")
define_option_string(MILVUS_WITH_ZLIB           "Using Zlib from"       "module" "package" "module")
define_option_string(MILVUS_WITH_NLOHMANN_JSON  "nlohmann json from"    "module" "package" "module")
define_option_string(MILVUS_WITH_GTEST          "Using GTest from"      "module" "package" "module")
define_option_string(MILVUS_WITH_EIGEN          "Using Eigen from"      "module" "package" "module")

set(BUILD_SCRIPTS_DIR ${PROJECT_SOURCE_DIR}/scripts)

# load third packages and milvus-proto
include(ThirdPartyPackages)
include(MilvusProtoGen)

# add testing
if (MILVUS_BUILD_TEST)
    if ( "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" AND MILVUS_BUILD_COVERAGE )
        # Set compile flag for code coverage.
        # Note: Only do this in Debug/unittest mode. Must do this before add_subdirectory(src).
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    endif ()
    enable_testing()
    add_subdirectory(test)
endif ()

add_subdirectory(src)
add_subdirectory(examples)

#
# make lint
#
find_package(Python)

if (NOT LINT_EXCLUSIONS_FILE)
    set(LINT_EXCLUSIONS_FILE ${BUILD_SCRIPTS_DIR}/lint_exclusions.txt)
endif ()

if (NOT FORMAT_EXCLUSIONS_FILE)
    set(FORMAT_EXCLUSIONS_FILE ${BUILD_SCRIPTS_DIR}/format_exclusions.txt)
endif ()

add_custom_target(lint
        ${Python_EXECUTABLE}
        ${BUILD_SCRIPTS_DIR}/run_cpplint.py
        --cpplint_binary
        ${BUILD_SCRIPTS_DIR}/cpplint.py
        --exclude_globs
        ${LINT_EXCLUSIONS_FILE}
        --source_dir
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${MILVUS_LINT_QUIET})


#
# clang-format targets
#
if (${CLANG_FORMAT_FOUND})
    # runs clang format and updates files in place.
    add_custom_target(clang-format
            ${Python_EXECUTABLE}
            ${BUILD_SCRIPTS_DIR}/run_clang_format.py
            --clang_format_binary
            ${CLANG_FORMAT_BIN}
            --exclude_globs
            ${FORMAT_EXCLUSIONS_FILE}
            --source_dir
            ${CMAKE_CURRENT_SOURCE_DIR}
            --fix
            ${MILVUS_LINT_QUIET})

    # runs clang format and exits with a non-zero exit code if any files need to be reformatted
    add_custom_target(check-clang-format
            ${Python_EXECUTABLE}
            ${BUILD_SCRIPTS_DIR}/run_clang_format.py
            --clang_format_binary
            ${CLANG_FORMAT_BIN}
            --exclude_globs
            ${FORMAT_EXCLUSIONS_FILE}
            --source_dir
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${MILVUS_LINT_QUIET})
endif ()

#
# clang-tidy targets
#
if (${CLANG_TIDY_FOUND})
    # runs clang-tidy and attempts to fix any warning automatically
    add_custom_target(clang-tidy
            ${Python_EXECUTABLE}
            ${BUILD_SCRIPTS_DIR}/run_clang_tidy.py
            --clang_tidy_binary
            ${CLANG_TIDY_BIN}
            --exclude_globs
            ${LINT_EXCLUSIONS_FILE}
            --compile_commands
            ${CMAKE_BINARY_DIR}/compile_commands.json
            --source_dir
            ${CMAKE_CURRENT_SOURCE_DIR}
            --fix
            ${MILVUS_LINT_QUIET})

    # runs clang-tidy and exits with a non-zero exit code if any errors are found.
    add_custom_target(check-clang-tidy
            ${Python_EXECUTABLE}
            ${BUILD_SCRIPTS_DIR}/run_clang_tidy.py
            --clang_tidy_binary
            ${CLANG_TIDY_BIN}
            --exclude_globs
            ${LINT_EXCLUSIONS_FILE}
            --compile_commands
            ${CMAKE_BINARY_DIR}/compile_commands.json
            --source_dir
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${MILVUS_LINT_QUIET})
endif ()

# output config summary
config_summary()
