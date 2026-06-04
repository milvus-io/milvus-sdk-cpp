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

cmake_minimum_required(VERSION 3.5)
project(milvus_sdk LANGUAGES CXX)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard used to build Milvus C++ SDK")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(BUILD_SCRIPTS_DIR ${PROJECT_SOURCE_DIR}/scripts)

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
define_option(MILVUS_BUILD_EXAMPLES "Build examples" ON)

if (DEFINED MILVUS_SDK_VERSION)
    message("MILVUS_SDK_VERSION = ${MILVUS_SDK_VERSION}")
else()
    # MILVUS_SDK_VERSION is defined in the scripts/build.sh, it could be undefined if user build the project with cmake command
    message("Reset MILVUS_SDK_VERSION by git command")
    execute_process(
        COMMAND git describe --tags --abbrev=0
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE MILVUS_SDK_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if (NOT MILVUS_SDK_VERSION)
        set(MILVUS_SDK_VERSION "Unknown")
        message("Git command failed, using fallback version: ${MILVUS_SDK_VERSION}")
    endif()
    message("MILVUS_SDK_VERSION = ${MILVUS_SDK_VERSION}")
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/impl/version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/src/impl/version.h @ONLY)

set_option_category("Thirdparty")
# use thirdparty from:
#  auto: Find -> Fetch_Content_Then_Find
#  package: Find
#  module: Fetch_Content_Then_Find
define_option_string(MILVUS_WITH_GRPC           "Using gRPC from"       "module" "package" "module")
define_option_string(MILVUS_WITH_GTEST          "Using GTest from"      "module" "package" "module")

# CMAKE_INSTALL_RPATH and CMAKE_INSTALL_RPATH_USE_LINK_PATH will set RPATH to all excutable TARGET
# self-installed dynamic libraries will be correctly linked by excutable
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if ("${BUILD_FROM_CONAN}" STREQUAL "ON")
    # Conan-managed dependencies: always use package-provided gRPC/GTest.
    set(MILVUS_WITH_GRPC "package")
    set(MILVUS_WITH_GTEST "package")

    # Do not embed Conan cache paths in installed libraries; Fedora RPM rejects them.
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

    # Dependencies are provided via find_package() (expected through Conan CMakeDeps).
    find_package(absl CONFIG REQUIRED)
    find_package(Protobuf CONFIG REQUIRED)
    find_package(gRPC CONFIG REQUIRED)

    find_package(nlohmann_json CONFIG REQUIRED)
else()
    # set the GRPC_PATH variable to specify external gRPC installed path
    if ("${GRPC_PATH}" STREQUAL "")
        set(MILVUS_WITH_GRPC "module")
        set(CMAKE_INSTALL_RPATH "/usr/lib" "${CMAKE_INSTALL_PREFIX}/lib" "${CMAKE_INSTALL_PREFIX}/lib64" )
    else ()
        set(MILVUS_WITH_GRPC "package")
        set(CMAKE_INSTALL_RPATH "/usr/lib" "${CMAKE_INSTALL_PREFIX}/lib" "${CMAKE_INSTALL_PREFIX}/lib64" "${GRPC_PATH}/lib" )
    endif ()

    # Non-Conan builds should also use a self-contained GTest (FetchContent) unless explicitly overridden.
    if ("${MILVUS_WITH_GTEST}" STREQUAL "")
        set(MILVUS_WITH_GTEST "module")
    endif()

    include(ThirdPartyPackages)
endif()

# load milvus-proto generator
include(MilvusProtoGen)

# add testing
if (MILVUS_BUILD_TEST)
    if ( "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" AND MILVUS_BUILD_COVERAGE )
        # Set compile flag for code coverage.
        # Note: Only do this in Debug/unittest mode. Must do this before add_subdirectory(src).
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    endif ()
    enable_testing()
endif ()

add_subdirectory(src)

if (MILVUS_BUILD_TEST)
    add_subdirectory(test)
endif ()

if (MILVUS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif ()

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

#
# Packaging (CPack)
#
# Detect the distro to pick the correct generator (DEB / RPM) and embed the
# distro name in the package filename, so a release tag produces artifacts
# like `libmilvus-2.6.2-x86_64-Ubuntu_20.04.deb` or
# `libmilvus-2.6.2-x86_64-Fedora_39.rpm`.
#
set(DISTRO_TAG "${CMAKE_SYSTEM_NAME}")
if (APPLE)
    execute_process(
        COMMAND sw_vers -productVersion
        OUTPUT_VARIABLE MACOS_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if (MACOS_VERSION)
        string(REGEX MATCH "^[0-9]+" MACOS_MAJOR_VERSION "${MACOS_VERSION}")
        if (MACOS_MAJOR_VERSION)
            set(DISTRO_TAG "macOS_${MACOS_MAJOR_VERSION}")
        else()
            set(DISTRO_TAG "macOS")
        endif()
    else()
        set(DISTRO_TAG "macOS")
    endif()
endif()
if (EXISTS "/etc/os-release")
    file(READ "/etc/os-release" _OS_RELEASE)
    # Match ID= and VERSION_ID= at start of line (after a newline *or* at the
    # very beginning of the file — some /etc/os-release variants don't have a
    # leading blank/comment line before these fields).
    if (_OS_RELEASE MATCHES "(^|\n)ID=\"?([^\"\n]+)\"?")
        set(DISTRO_ID "${CMAKE_MATCH_2}")
    endif()
    if (_OS_RELEASE MATCHES "(^|\n)VERSION_ID=\"?([^\"\n]+)\"?")
        set(DISTRO_VERSION "${CMAKE_MATCH_2}")
    endif()
    if (DISTRO_ID AND DISTRO_VERSION)
        # capitalise first letter: ubuntu -> Ubuntu, fedora -> Fedora
        string(SUBSTRING "${DISTRO_ID}" 0 1 _FIRST)
        string(TOUPPER "${_FIRST}" _FIRST_UPPER)
        string(SUBSTRING "${DISTRO_ID}" 1 -1 _REST)
        set(DISTRO_TAG "${_FIRST_UPPER}${_REST}_${DISTRO_VERSION}")
    endif()
endif()

# Normalize the version for packaging: strip a leading "v" so both
# `git describe` output (e.g. "v2.6.2") and explicit values (e.g. "2.6.2")
# produce the same DEB/RPM version string, which must be digits-and-dots to
# satisfy Debian/RPM policy. Keep MILVUS_SDK_VERSION untouched so it still
# reflects the raw version string in logs and version.h.
string(REGEX REPLACE "^v" "" MILVUS_PACKAGE_VERSION "${MILVUS_SDK_VERSION}")

set(CPACK_PACKAGE_NAME "libmilvus")
set(CPACK_PACKAGE_VENDOR "Milvus")
set(CPACK_PACKAGE_VERSION "${MILVUS_PACKAGE_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Milvus C++ SDK")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/milvus-io/milvus-sdk-cpp")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_BINARY_DIR}/Pack")
set(CPACK_PACKAGE_FILE_NAME
    "libmilvus-${MILVUS_PACKAGE_VERSION}-${CMAKE_SYSTEM_PROCESSOR}-${DISTRO_TAG}")

if (DISTRO_ID STREQUAL "ubuntu" OR DISTRO_ID STREQUAL "debian")
    set(CPACK_GENERATOR "DEB")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Milvus <maintainer@milvus.io>")
    set(CPACK_DEBIAN_PACKAGE_SECTION "libs")
elseif (DISTRO_ID STREQUAL "fedora" OR DISTRO_ID STREQUAL "rhel" OR DISTRO_ID STREQUAL "centos")
    set(CPACK_GENERATOR "RPM")
    set(CPACK_RPM_PACKAGE_LICENSE "Apache-2.0")
    set(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")
else()
    set(CPACK_GENERATOR "TGZ")
endif()

include(CPack)
