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

include(CPM)

# grpc
if ("${MILVUS_WITH_GRPC}" STREQUAL "pakcage")
    find_package(grpc REQUIRED)
else ()
    if (WIN32)
        set(OPENSSL_NO_ASM_TXT "YES")
    else ()
        set(OPENSSL_NO_ASM_TXT "NO")
    endif ()
    CPMAddPackage(
        NAME grpc
        VERSION 1.49.1
        GITHUB_REPOSITORY grpc/grpc
        EXCLUDE_FROM_ALL YES
        OPTIONS
            "gRPC_SSL_PROVIDER module"
            "gRPC_PROTOBUF_PROVIDER module"
            "gRPC_BUILD_TESTS OFF"
            "RE2_BUILD_TESTING OFF"
            "ABSL_PROPAGATE_CXX_STD ON"
            "OPENSSL_NO_ASM ${OPENSSL_NO_ASM_TXT}"
    )
    if (grpc_ADDED)
        add_library(gRPC::grpc++ ALIAS grpc++)
    endif ()
endif ()


# nlohmann_json
if ("${MILVUS_WITH_NLOHMANN_JSON}" STREQUAL "package")
    find_package(nlohmann_json REQUIRED)
else ()
    CPMAddPackage(
        NAME nlohmann_json
        VERSION 3.11.2
        GITHUB_REPOSITORY nlohmann/json
    )
endif ()
