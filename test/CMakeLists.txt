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

find_package(GTest)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/proto-gen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src/impl)

if (${GTest_FOUND})
    message(STATUS "Using gtest from system")
    include_directories(${GTEST_INCLUDE_DIRS})
else ()
    message(STATUS "Using gtest from source")
    if(NOT googletest_POPULATED)
        FetchContent_Populate(googletest)
        add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()

    set(GTEST_LIBRARIES gtest)
    set(GMOCK_LIBRARIES gmock)
    set(GTEST_MAIN_LIBRARIES gtest_main)
endif ()

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/ut ut_files)
add_executable(testing-ut ${ut_files})
target_link_libraries(testing-ut PRIVATE milvus_sdk ${GTEST_LIBRARIES} ${GMOCK_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)
gtest_discover_tests(testing-ut)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/it it_files)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/it/mocks it_mocks_files)
add_executable(testing-it ${it_files} ${it_mocks_files})
target_link_libraries(testing-it PRIVATE milvus_sdk ${GTEST_LIBRARIES} ${GMOCK_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)
gtest_discover_tests(testing-it)

# st only available under linux
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/st st_files)
add_executable(testing-st ${st_files})
target_link_libraries(testing-st PRIVATE milvus_sdk ${GTEST_LIBRARIES} ${GMOCK_LIBRARIES} ${GTEST_MAIN_LIBRARIES} pthread)
add_custom_target(system-test
    COMMAND $<TARGET_FILE:testing-st>
    DEPENDS testing-st
)
endif(CMAKE_SYSTEM_NAME MATCHES "Linux")
