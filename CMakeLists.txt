cmake_minimum_required(VERSION 3.20)
project(slm-porting CXX)


enable_testing()


find_package(Python3 REQUIRED COMPONENTS Interpreter)


# the python installation should be able to load the python package "slmporting"
set(CMAKE_DIR "${CMAKE_SOURCE_DIR}/cmake")
set(PYTHON_DEPENDENCY_CHECK_SCRIPT "${CMAKE_DIR}/dependency_check.py")
if (NOT EXISTS ${PYTHON_DEPENDENCY_CHECK_SCRIPT})
	message(FATAL_ERROR "${PYTHON_DEPENDENCY_CHECK_SCRIPT} was not found")
endif ()

execute_process(
	COMMAND ${Python3_EXECUTABLE} ${PYTHON_DEPENDENCY_CHECK_SCRIPT}
	WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
	RESULT_VARIABLE dependency_check_result
	OUTPUT_VARIABLE dependency_check_output
	ERROR_VARIABLE  dependency_check_error
)

if (NOT dependency_check_result EQUAL 0)
	message(STATUS "${dependency_check_output}")
	message(FATAL_ERROR "There are missing dependencies in the python section, please refer to pyproject.toml")
else ()
	message(STATUS "Python dependencies satisfied")
endif ()


# I propose a root level cmake for coordinating the interaction
# between the python section and the c++ one. I want the python
# scripts to be able to access the c++ executables for verification
# and profiling purposes.
add_subdirectory("${CMAKE_SOURCE_DIR}/cpp")


# Here I want custom rules for invoking the verification and profiling scripts
