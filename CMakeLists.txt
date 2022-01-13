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
project(milvus_sdk LANGUAGES CXX C)

set(CMAKE_VERBOSE_MAKEFILE OFF)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

include(FindPython2)
include(DefineOptions)
include(BuildUtils)
include(FindClangTools)
include(ThirdPartyPackages)
include(GoogleTest)
include(CTest)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(BUILD_SCRIPTS_DIR ${CMAKE_SOURCE_DIR}/scripts)


unset(CMAKE_EXPORT_COMPILE_COMMANDS CACHE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (BUILD_TEST STREQUAL "ON")
    if ( CMAKE_BUILD_TYPE STREQUAL "Debug" )
        # Set compile flag for code coverage.
        # Note: Only do this in Debug/unittest mode. Must do this before add_subdirectory(src).
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    endif ()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DELPP_DISABLE_LOGS")
    enable_testing()
    add_subdirectory(test)
endif ()

add_subdirectory(src)
add_subdirectory(examples)

# lint verbose options
if (NOT MILVUS_VERBOSE_LINT)
    set(MILVUS_LINT_QUIET "--quiet")
endif ()

#
# make lint
#
find_package(Python2)

if (NOT LINT_EXCLUSIONS_FILE)
    set(LINT_EXCLUSIONS_FILE ${BUILD_SCRIPTS_DIR}/lint_exclusions.txt)
endif ()

if (NOT FORMAT_EXCLUSIONS_FILE)
    set(FORMAT_EXCLUSIONS_FILE ${BUILD_SCRIPTS_DIR}/format_exclusions.txt)
endif ()

add_custom_target(lint
        ${Python2_EXECUTABLE}
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
            ${Python2_EXECUTABLE}
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
            ${Python2_EXECUTABLE}
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
            ${Python2_EXECUTABLE}
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
            ${Python2_EXECUTABLE}
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
