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

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/impl impl_files)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/impl/types impl_types_files)

add_library(milvus_sdk ${impl_files} ${impl_types_files})
# add proto gens
add_milvus_protos(milvus_sdk)
set_target_properties(milvus_sdk PROPERTIES OUTPUT_NAME milvus_sdk)
target_link_libraries(milvus_sdk gRPC::grpc++ nlohmann_json::nlohmann_json)
target_include_directories(milvus_sdk PUBLIC include)


# install
include(GNUInstallDirs)

# install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")

# install libs
install(TARGETS milvus_sdk
        EXPORT milvus_sdk
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})