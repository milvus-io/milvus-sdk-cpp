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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/impl)
include_directories(${milvus_proto_BINARY_DIR})

# resolve gtest
if ("${MILVUS_WITH_GTEST}" STREQUAL "pakcage")
    find_package(GTest REQUIRED)
else ()
    CPMAddPackage(
        NAME googletest
        GIT_TAG release-1.12.1
        VERSION 1.12.1
        GITHUB_REPOSITORY google/googletest
        OPTIONS
            "INSTALL_GTEST OFF"
            "gtest_force_shared_crt ON"
    )
endif ()

set(GTEST_LIBRARIES gtest)
set(GMOCK_LIBRARIES gmock)
set(GTEST_MAIN_LIBRARIES gtest_main)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/ut ut_files)
add_executable(testing-ut ${ut_files})
target_compile_options(testing-ut PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
target_link_libraries(testing-ut PRIVATE milvus_sdk ${GTEST_LIBRARIES} ${GMOCK_LIBRARIES} ${GTEST_MAIN_LIBRARIES})

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/it it_files)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/it/mocks it_mocks_files)
add_executable(testing-it ${it_files} ${it_mocks_files})
target_compile_options(testing-it PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
target_link_libraries(testing-it PRIVATE milvus_sdk ${GTEST_LIBRARIES} ${GMOCK_LIBRARIES} ${GTEST_MAIN_LIBRARIES})

# st only available under linux/macos
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/st st_files)
add_executable(testing-st ${st_files})
target_link_libraries(testing-st PRIVATE milvus_sdk ${GTEST_LIBRARIES} ${GMOCK_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
endif()
