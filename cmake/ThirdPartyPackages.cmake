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

# grpc
FetchContent_Declare(
    grpc
    GIT_REPOSITORY    https://github.com/grpc/grpc.git
    GIT_TAG           v${GRPC_VERSION}
    GIT_SHALLOW       TRUE
)

# nlohmann_json
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY    https://github.com/nlohmann/json.git
    GIT_TAG           v${NLOHMANN_JSON_VERSION}
    GIT_SHALLOW       TRUE
)

# googletest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY    https://github.com/google/googletest.git
    GIT_TAG           release-${GOOGLETEST_VERSION}
    GIT_SHALLOW       TRUE
)

FetchContent_Declare(
    eigen3
    GIT_REPOSITORY    https://gitlab.com/libeigen/eigen.git
    GIT_TAG           3.4.0
    GIT_SHALLOW       TRUE
)

# grpc
if ("${MILVUS_WITH_GRPC}" STREQUAL "pakcage")
    find_package(grpc REQUIRED)
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

# eigen3
if ("${MILVUS_WITH_EIGEN}" STREQUAL "package")
    find_package(Eigen3 REQUIRED NO_MODULE)
else ()
    if (NOT eigen3_POPULATED)
        FetchContent_Populate(eigen3)
        set(BUILD_TESTING OFF CACHE INTERNAL "")
        add_subdirectory(${eigen3_SOURCE_DIR} ${eigen3_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif ()
endif ()
