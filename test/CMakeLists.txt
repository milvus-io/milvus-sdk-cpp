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

# GTest is provided in different ways depending on the build mode:
# - BUILD_FROM_CONAN=ON: Conan CMakeDeps provides GTestConfig.cmake and imported targets.
# - non-Conan builds (scripts/build.sh -z): we rely on ThirdPartyPackages.cmake to fetch/build googletest.
if ("${BUILD_FROM_CONAN}" STREQUAL "ON")
    find_package(GTest CONFIG REQUIRED)
endif()

set(GTEST_LIBRARIES GTest::gtest)
set(GMOCK_LIBRARIES GTest::gmock)
set(GTEST_MAIN_LIBRARIES GTest::gtest_main)

set(UT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ut")
file(GLOB_RECURSE ut_files
    "${UT_DIR}/*.cpp"
    "${UT_DIR}/*.cxx"
    "${UT_DIR}/*.cc"
)
add_executable(testing-ut ${ut_files})
target_compile_options(testing-ut PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
target_link_libraries(testing-ut PRIVATE milvus_sdk ${GTEST_LIBRARIES} ${GMOCK_LIBRARIES} ${GTEST_MAIN_LIBRARIES})

set(IT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/it")
file(GLOB_RECURSE it_files
    "${IT_DIR}/*.cpp"
    "${IT_DIR}/*.cxx"
    "${IT_DIR}/*.cc"
)
add_executable(testing-it ${it_files})
target_compile_options(testing-it PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
target_link_libraries(testing-it PRIVATE milvus_sdk ${GTEST_LIBRARIES} ${GMOCK_LIBRARIES} ${GTEST_MAIN_LIBRARIES})

# st only available under linux/macos
if (CMAKE_SYSTEM_NAME MATCHES "(Linux|Darwin)")
set(ST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/st")
file(GLOB_RECURSE st_files
    "${ST_DIR}/*.cpp"
    "${ST_DIR}/*.cxx"
    "${ST_DIR}/*.cc"
)
add_executable(testing-st ${st_files})
target_link_libraries(testing-st PRIVATE milvus_sdk ${GTEST_LIBRARIES} ${GMOCK_LIBRARIES})

if (MILVUS_WITH_TESTCONTAINERS)
    include(FetchContent)
    set(TESTCONTAINERS_ROOT "" CACHE PATH "Path to a testcontainers-native install prefix")

    if (TESTCONTAINERS_ROOT)
        find_path(TESTCONTAINERS_INCLUDE_DIR
            NAMES testcontainers-c.h testcontainers-c/container.h
            HINTS ${TESTCONTAINERS_ROOT}/include)
        find_library(TESTCONTAINERS_LIBRARY
            NAMES testcontainers-c
            HINTS ${TESTCONTAINERS_ROOT}/lib ${TESTCONTAINERS_ROOT}/lib64)
        set(TESTCONTAINERS_RPATH "${TESTCONTAINERS_ROOT}/lib;${TESTCONTAINERS_ROOT}/lib64")
    elseif (CMAKE_VERSION VERSION_GREATER_EQUAL "3.26")
        FetchContent_Declare(
            testcontainers_native
            GIT_REPOSITORY https://github.com/testcontainers/testcontainers-native.git
            GIT_TAG v0.1.0
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE
        )
        FetchContent_MakeAvailable(testcontainers_native)
        set(TESTCONTAINERS_INCLUDE_DIR
            ${testcontainers_native_BINARY_DIR}/testcontainers-c
            CACHE PATH "Testcontainers Native include directory" FORCE)
        set(TESTCONTAINERS_LIBRARY
            testcontainers-c
            CACHE STRING "Testcontainers Native library target" FORCE)
        set(TESTCONTAINERS_RPATH "${testcontainers_native_BINARY_DIR}/testcontainers-c")
    else()
        message(WARNING "Testcontainers requires CMake 3.26+ to build from source. Set TESTCONTAINERS_ROOT to use a prebuilt install.")
    endif()

    if (TESTCONTAINERS_INCLUDE_DIR AND TESTCONTAINERS_LIBRARY)
        target_include_directories(testing-st PRIVATE ${TESTCONTAINERS_INCLUDE_DIR})
        target_link_libraries(testing-st PRIVATE ${TESTCONTAINERS_LIBRARY})
        target_compile_definitions(testing-st PRIVATE MILVUS_WITH_TESTCONTAINERS=1)
        if (TESTCONTAINERS_RPATH)
            set_target_properties(testing-st PROPERTIES BUILD_RPATH "${TESTCONTAINERS_RPATH}")
            foreach(testcontainers_rpath IN LISTS TESTCONTAINERS_RPATH)
                if (testcontainers_rpath)
                    target_link_options(testing-st PRIVATE
                        $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wl,-rpath,${testcontainers_rpath}>)
                endif()
            endforeach()
        endif()
    else()
        message(WARNING "Testcontainers not found; system tests will use localhost:19530.")
    endif()
endif()
endif()
