cmake_minimum_required(VERSION 3.20)
project(slm-porting CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

enable_testing()


set(SRC_DIR   "${CMAKE_SOURCE_DIR}/src")
set(IMPL_DIR  "${CMAKE_SOURCE_DIR}/impl")
set(INCL_DIR  "${CMAKE_SOURCE_DIR}/include")
set(TESTS_DIR "${CMAKE_SOURCE_DIR}/tests")
set(TESTS_EXE "")
file(GLOB TESTS "${TESTS_DIR}/*")


add_library(common STATIC
	"${CMAKE_SOURCE_DIR}/src/units.cpp"
	"${CMAKE_SOURCE_DIR}/src/slm.cpp"
	"${CMAKE_SOURCE_DIR}/src/rand.cpp"
)
include_directories(common "${CMAKE_SOURCE_DIR}/include")


foreach (TEST_FILE ${TESTS})
	get_filename_component(TEST_NAME ${TEST_FILE} NAME_WLE)

	add_executable(${TEST_NAME} ${TEST_FILE})
	target_include_directories(${TEST_NAME} PRIVATE ${INCL_DIR})
	target_link_libraries(${TEST_NAME} PRIVATE common)

	add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
	list(APPEND TESTS_EXE ${TEST_NAME})
endforeach()


# this forces the tests to be actual targets
# they get recompiled before invoking the tests
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS ${TESTS_EXE})
