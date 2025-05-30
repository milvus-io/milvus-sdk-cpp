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

set(GRPC_VERSION 1.49.1)
set(NLOHMANN_JSON_VERSION 3.11.3)
set(GOOGLETEST_VERSION 1.12.1)
set(CPP_HTTPLIB_VERSION 0.18.5)
set(BROTLI_VERSION 1.1.0)
Set(FETCHCONTENT_QUIET FALSE)

# grpc
FetchContent_Declare(
    grpc
    GIT_REPOSITORY    https://github.com/grpc/grpc.git
    GIT_TAG           v${GRPC_VERSION}
    GIT_SHALLOW       TRUE
    GIT_PROGRESS      TRUE
)

# nlohmann_json
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY    https://github.com/nlohmann/json.git
    GIT_TAG           v${NLOHMANN_JSON_VERSION}
    GIT_SHALLOW       TRUE
    GIT_PROGRESS      TRUE
)

# googletest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY    https://github.com/google/googletest.git
    GIT_TAG           release-${GOOGLETEST_VERSION}
    GIT_SHALLOW       TRUE
    GIT_PROGRESS      TRUE
)

# cpp-httplib
FetchContent_Declare(
    cpp-httplib
    GIT_REPOSITORY    https://github.com/yhirose/cpp-httplib.git
    GIT_TAG           v${CPP_HTTPLIB_VERSION}
    GIT_SHALLOW       TRUE
    GIT_PROGRESS      TRUE
)

# brotli
FetchContent_Declare(
    brotli
    GIT_REPOSITORY    https://github.com/google/brotli.git
    GIT_TAG           v${BROTLI_VERSION}
    GIT_SHALLOW       TRUE
    GIT_PROGRESS      TRUE
)

# grpc
if ("${MILVUS_WITH_GRPC}" STREQUAL "package")
    find_package(Protobuf REQUIRED)
    find_package(gRPC REQUIRED)
else ()
    if (WIN32)
        set(OPENSSL_NO_ASM_TXT "YES")
    else ()
        set(OPENSSL_NO_ASM_TXT "NO")
    endif ()
    if (NOT grpc_POPULATED)
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
        add_library(gRPC::grpc++ ALIAS grpc++)
        add_executable(gRPC::grpc_cpp_plugin ALIAS grpc_cpp_plugin)
    endif ()
endif ()


# nlohmann_json
if ("${MILVUS_WITH_NLOHMANN_JSON}" STREQUAL "package")
    find_package(nlohmann_json REQUIRED)
else ()
    if (NOT nlohmann_json_POPULATED)
        FetchContent_Populate(nlohmann_json)
        add_subdirectory(${nlohmann_json_SOURCE_DIR} ${nlohmann_json_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif ()
endif ()

# cpp-httplib
if ("${MILVUS_WITH_CPP_HTTPLIB}" STREQUAL "package")
    find_package(cpp-httplib REQUIRED)
else ()
    if (NOT cpp-httplib_POPULATED)
        FetchContent_Populate(cpp-httplib)
        add_subdirectory(${cpp-httplib_SOURCE_DIR} ${cpp-httplib_BINARY_DIR} EXCLUDE_FROM_ALL)
        include_directories(${cpp-httplib_SOURCE_DIR})
    endif ()
endif ()

# brotli
if ("${MILVUS_WITH_BROTLI}" STREQUAL "package")
    find_package(Brotli REQUIRED)
else ()
    if (NOT brotli_POPULATED)
        FetchContent_Populate(brotli)
        add_subdirectory(${brotli_SOURCE_DIR} ${brotli_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif ()
endif ()
