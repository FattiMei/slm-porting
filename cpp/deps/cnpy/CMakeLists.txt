find_package(ZLIB REQUIRED)


# this is the .npz library we export to the cpp module
add_library(cnpy STATIC cnpy.cpp)
target_include_directories(cnpy
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
	PRIVATE ${ZLIB_INCLUDE_DIRS}
)
target_link_libraries(cnpy ${ZLIB_LIBRARIES})


set(EXAMPLE_EXE example_c_to_npz)
add_executable(${EXAMPLE_EXE} test_c_to_npz.cpp)
target_include_directories(${EXAMPLE_EXE} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${EXAMPLE_EXE} PRIVATE cnpy)


# we can't supply multiple commands to the "add_test" function
set(VERIFICATION_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/test_c_to_npz.py")
set(EXAMPLE_FILENAME "output.npz")
add_custom_target(${EXAMPLE_FILENAME}
	COMMAND ${EXAMPLE_EXE} ${EXAMPLE_FILENAME}
	DEPENDS ${EXAMPLE_EXE}
)

add_test(
	NAME ${EXAMPLE_FILENAME}
	COMMAND ${Python3_EXECUTABLE} ${VERIFICATION_SCRIPT} ${EXAMPLE_FILENAME}
)
