cmake_minimum_required(VERSION 3.0.0)
project(CubSpMV2CUSPARSE LANGUAGES CUDA)

set(CMAKE_CUDA_ARCHITECTURES "86")
add_executable(CubSpMV2CUSPARSE main.cu)
target_link_libraries(CubSpMV2CUSPARSE PRIVATE cusparse)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
include(CPack)
