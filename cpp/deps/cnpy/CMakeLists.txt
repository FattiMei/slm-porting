cmake_minimum_required(VERSION 3.20)

find_package(ZLIB REQUIRED)

add_library(cnpy STATIC cnpy.cpp)
target_include_directories(cnpy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(cnpy PRIVATE ${ZLIB_INCLUDE_DIRS})
target_link_libraries(cnpy ${ZLIB_LIBRARIES})
