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


include(FetchContent)


# Ensure that a default make is set
if ("${MAKE}" STREQUAL "")
    find_program(MAKE make)
endif ()

if (NOT DEFINED MAKE_BUILD_ARGS)
    set(MAKE_BUILD_ARGS "-j8")
endif ()
message(STATUS "Third Party MAKE_BUILD_ARGS = ${MAKE_BUILD_ARGS}")

# ----------------------------------------------------------------------
# Find pthreads

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ----------------------------------------------------------------------
# External source default urls

if (DEFINED ENV{MILVUS_GRPC_URL})
    set(GRPC_SOURCE_URL "$ENV{MILVUS_GRPC_URL}")
else ()
    set(GRPC_SOURCE_URL
            "https://github.com/milvus-io/grpc-milvus/archive/master.zip")
endif ()

if (DEFINED ENV{MILVUS_GTEST_URL})
    set(GTEST_SOURCE_URL "$ENV{MILVUS_GTEST_URL}")
else ()
    set(GTEST_SOURCE_URL "https://github.com/google/googletest/archive/release-1.10.0.tar.gz")
endif ()


# Openssl required for grpc
if (CMAKE_HOST_APPLE)
    execute_process(
        COMMAND brew --prefix openssl@3
        OUTPUT_VARIABLE USER_OPENSSL_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY
    )
    set(OPENSSL_ROOT_DIR ${USER_OPENSSL_PATH})
endif (CMAKE_HOST_APPLE)

find_package(OpenSSL REQUIRED)


# grpc
FetchContent_Declare(
    grpc
    URL ${GRPC_SOURCE_URL}
)

FetchContent_Declare(
    googletest
    URL ${GTEST_SOURCE_URL}
)

# enable grpc
if(NOT grpc_POPULATED)
    FetchContent_Populate(grpc)
    set(gRPC_SSL_PROVIDER "package" CACHE INTERNAL "Provider of ssl library")
    add_subdirectory(${grpc_SOURCE_DIR} ${grpc_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
