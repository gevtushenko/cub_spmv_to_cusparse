cmake_minimum_required(VERSION 3.0.0)
project(CubSpMV2CUSPARSE VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(CubSpMV2CUSPARSE main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
