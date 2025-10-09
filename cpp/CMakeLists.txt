cmake_minimum_required(VERSION 3.20)
project(slm-porting CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if (NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
	set(AGGRESSIVE_OPTIMIZATIONS "-march=native" "-ftree-vectorize")
endif()

enable_testing()


set(SRC_DIR   "${CMAKE_SOURCE_DIR}/src")
set(IMPL_DIR  "${CMAKE_SOURCE_DIR}/impl")
set(INCL_DIR  "${CMAKE_SOURCE_DIR}/include")
set(TESTS_DIR "${CMAKE_SOURCE_DIR}/tests")
file(GLOB TESTS "${TESTS_DIR}/*")


add_library(common STATIC
	"${CMAKE_SOURCE_DIR}/src/units.cpp"
)
include_directories(common "${CMAKE_SOURCE_DIR}/include")


foreach (TEST_FILE ${TESTS})
	get_filename_component(TEST_NAME ${TEST_FILE} NAME_WLE)

	add_executable(${TEST_NAME} ${TEST_FILE})
	target_include_directories(${TEST_NAME} PRIVATE ${INCL_DIR})
	target_link_libraries(${TEST_NAME} PRIVATE common)

	add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
