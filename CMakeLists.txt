cmake_minimum_required(VERSION 3.12)
project(milvus_sdk LANGUAGES CXX C)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(BuildUtils)
include(DefineOptions)
include(ThirdPartyPackages)

