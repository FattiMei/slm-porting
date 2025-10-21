set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# `cnpy` is the exported static library for
# reading and writing .npz files from c++
add_subdirectory(deps/cnpy)

# find_package(OpenMP REQUIRED)
# 
# message(STATUS "Hello, world!")


# add_subdirectory("${CMAKE_CURRENT_DIR}/deps/cnpy")
# 
# enable_testing()
# 
# 
# if (NOT CMAKE_BUILD_TYPE)
# 	message("No build type specified, defaulting to Release")
# 	set(CMAKE_BUILD_TYPE Release)
# endif ()
# 
# set(AGGRESSIVE_OPTIMIZATIONS "-march=native" "-ftree-vectorize" "-ffast-math")
# 
# set(SRC_DIR   "${CMAKE_SOURCE_DIR}/src")
# set(IMPL_DIR  "${CMAKE_SOURCE_DIR}/impl")
# set(INCL_DIR  "${CMAKE_SOURCE_DIR}/include")
# set(VERIFICATION_EXE "cpp-verification")
# set(TESTS_DIR "${CMAKE_SOURCE_DIR}/tests")
# set(TESTS_EXE "")
# file(GLOB TESTS "${TESTS_DIR}/*")
# 
# 
# add_library(common STATIC
# 	"${CMAKE_SOURCE_DIR}/src/units.cpp"
# 	"${CMAKE_SOURCE_DIR}/src/slm.cpp"
# 	"${CMAKE_SOURCE_DIR}/src/rand.cpp"
# )
# include_directories(common "${CMAKE_SOURCE_DIR}/include")
# 
# 
# add_executable(cpp-verification "${CMAKE_SOURCE_DIR}/src/verification.cpp")
# target_include_directories(cpp-verification PRIVATE ${INCL_DIR})
# target_link_libraries(${VERIFICATION_EXE} PRIVATE common)
# target_link_libraries(${VERIFICATION_EXE} PRIVATE cnpy)
# target_link_libraries(${VERIFICATION_EXE} PRIVATE OpenMP::OpenMP_CXX)
# target_compile_options(${VERIFICATION_EXE} PRIVATE ${AGGRESSIVE_OPTIMIZATIONS})
# 
# 
# foreach (TEST_FILE ${TESTS})
# 	get_filename_component(TEST_NAME ${TEST_FILE} NAME_WLE)
# 
# 	add_executable(${TEST_NAME} ${TEST_FILE})
# 	target_include_directories(${TEST_NAME} PRIVATE ${INCL_DIR})
# 	target_link_libraries(${TEST_NAME} PRIVATE common)
# 	target_link_libraries(${TEST_NAME} PRIVATE cnpy)
# 	target_link_libraries(${TEST_NAME} PRIVATE OpenMP::OpenMP_CXX)
# 	target_compile_options(${TEST_NAME} PRIVATE ${AGGRESSIVE_OPTIMIZATIONS})
# 
# 	add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
# 	list(APPEND TESTS_EXE ${TEST_NAME})
# endforeach()
# 
# 
# # this forces the tests to be actual targets
# # they get recompiled before invoking the tests
# add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS ${TESTS_EXE})
