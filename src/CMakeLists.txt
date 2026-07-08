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

set(IMPL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/impl")
file(GLOB_RECURSE impl_files
    "${IMPL_DIR}/*.cpp"
    "${IMPL_DIR}/*.cxx"
    "${IMPL_DIR}/*.cc"
)

add_library(milvus_sdk_objects OBJECT ${impl_files})
set_target_properties(milvus_sdk_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)
if (WIN32 AND BUILD_SHARED_LIBS)
    target_compile_definitions(milvus_sdk_objects PRIVATE MILVUS_SDK_SHARED MILVUS_SDK_BUILDING_LIBRARY)
endif()

# add proto gens
add_milvus_protos(milvus_sdk_objects)
target_link_libraries(milvus_sdk_objects PRIVATE protobuf::libprotobuf gRPC::grpc++)
target_compile_options(milvus_sdk_objects PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
target_include_directories(milvus_sdk_objects PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(milvus_sdk_objects PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty)

# precompile heavy headers to speed up compilation
target_precompile_headers(milvus_sdk_objects PRIVATE
    <string>
    <vector>
    <map>
    <memory>
    <unordered_map>
    <milvus/thirdparty/nlohmann/json.hpp>
    "${PROTO_BINARY_DIR}/common.pb.h"
    "${PROTO_BINARY_DIR}/schema.pb.h"
    "${PROTO_BINARY_DIR}/milvus.pb.h"
    "${PROTO_BINARY_DIR}/milvus.grpc.pb.h"
)

add_library(milvus_sdk $<TARGET_OBJECTS:milvus_sdk_objects>)
if (WIN32 AND BUILD_SHARED_LIBS)
    target_compile_definitions(milvus_sdk INTERFACE MILVUS_SDK_SHARED)
endif()
set_target_properties(milvus_sdk PROPERTIES OUTPUT_NAME milvus_sdk)
target_link_libraries(milvus_sdk PRIVATE gRPC::grpc++)
if (MILVUS_EXPORT_STATIC_DEPS)
    target_link_libraries(milvus_sdk PUBLIC
            $<LINK_ONLY:protobuf::libprotobuf>
            $<LINK_ONLY:gRPC::grpc++>)
endif()
target_include_directories(milvus_sdk PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(milvus_sdk PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty)

# install
include(GNUInstallDirs)

# install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp")

# install libs
install(TARGETS milvus_sdk
        EXPORT milvus_sdk
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

if (MILVUS_INSTALL_CMAKE_PACKAGE)
    install(EXPORT milvus_sdk
            FILE milvus_sdkTargets.cmake
            NAMESPACE milvus_sdk::
            DESTINATION ${MILVUS_SDK_CMAKE_DIR})

    install(FILES
            ${PROJECT_BINARY_DIR}/milvus_sdkConfig.cmake
            ${PROJECT_BINARY_DIR}/milvus_sdkConfigVersion.cmake
            DESTINATION ${MILVUS_SDK_CMAKE_DIR})
endif()
