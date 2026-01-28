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

include_guard(GLOBAL)

include(FetchContent)

# milvus server v2.4.23 is using grpc v1.59.0
set(GRPC_VERSION 1.65.0)
set(GOOGLETEST_VERSION 1.12.1)
Set(FETCHCONTENT_QUIET FALSE)

set(GRPC_SRC_URL https://github.com/grpc/grpc.git)
set(GTEST_SRC_URL https://github.com/google/googletest.git)

# grpc
FetchContent_Declare(
    grpc
    GIT_REPOSITORY    ${GRPC_SRC_URL}
    GIT_TAG           v${GRPC_VERSION}
    GIT_SHALLOW       TRUE
    GIT_PROGRESS      TRUE
)

# googletest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY    ${GTEST_SRC_URL}
    GIT_TAG           release-${GOOGLETEST_VERSION}
    GIT_SHALLOW       TRUE
    GIT_PROGRESS      TRUE
)

# grpc
if ("${MILVUS_WITH_GRPC}" STREQUAL "package")
    message(STATUS "Finding gRPC lib from specified path: ${GRPC_PATH}")
    find_package(absl
        CONFIG
        HINTS ${GRPC_PATH}
        REQUIRED)
    find_package(Protobuf
        CONFIG
        HINTS ${GRPC_PATH}
        REQUIRED)
    find_package(gRPC
        CONFIG
        HINTS ${GRPC_PATH}
        REQUIRED)
    set(protobuf_SOURCE_DIR ${GRPC_PATH}/include)
else ()
    if (WIN32)
        set(OPENSSL_NO_ASM_TXT "YES")
    else ()
        set(OPENSSL_NO_ASM_TXT "NO")
    endif ()
    if (NOT grpc_POPULATED)
        message(STATUS "Downloading gRPC source code from: ${GRPC_SRC_URL}")
        FetchContent_Populate(grpc)
        if (WIN32)
            set(OPENSSL_NO_ASM YES  CACHE INTERNAL "")
        endif()
        set(gRPC_SSL_PROVIDER "module" CACHE INTERNAL "")
        set(gRPC_PROTOBUF_PROVIDER "module" CACHE INTERNAL "")
        set(gRPC_BUILD_TESTS OFF CACHE INTERNAL "")
        set(RE2_BUILD_TESTING OFF CACHE INTERNAL "")
        set(ABSL_ENABLE_INSTALL ON CACHE INTERNAL "")
        set(ABSL_PROPAGATE_CXX_STD ON CACHE INTERNAL "")
        add_subdirectory(${grpc_SOURCE_DIR} ${grpc_BINARY_DIR} EXCLUDE_FROM_ALL)
        set(protobuf_SOURCE_DIR ${grpc_SOURCE_DIR}/third_party/protobuf/src)
        add_library(gRPC::grpc++ ALIAS grpc++)
        add_executable(gRPC::grpc_cpp_plugin ALIAS grpc_cpp_plugin)
    endif ()
endif ()
