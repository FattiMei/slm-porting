cmake_minimum_required(VERSION 3.20)
project(slm-porting CXX)


find_package(Python3 REQUIRED COMPONENTS Interpreter Development)


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
	OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NOT dependency_check_result EQUAL 0)
	message(STATUS "${dependency_check_error}")
	message(FATAL_ERROR "There are missing dependencies in the python section, please refer to pyproject.toml")
else ()
	message(STATUS "Python dependencies satisfied")
endif ()


# from the python installation I should be able to retrieve the Torch
# libraries for the static compilation of the cpp implementations
#
# it is mandatory to precompile the implementations, even if this
# allows running the python scripts with outdated sources because
# the compilation is VERY slow
# BONUS: by using a separate compilation path we can more easily
# iterate on the compiler errors and use LSP/Intellisense
set(PYTHON_TORCH_CONFIG_SCRIPT "${CMAKE_DIR}/torch_cmake_prefix_exporter.py")
execute_process(
	COMMAND ${Python3_EXECUTABLE} ${PYTHON_TORCH_CONFIG_SCRIPT}
	WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
	RESULT_VARIABLE torch_cmake_prefix_path_result
	OUTPUT_VARIABLE torch_cmake_prefix_path_output
	ERROR_VARIABLE torch_cmake_prefix_path_error
	OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NOT torch_cmake_prefix_path_result EQUAL 0)
	message(STATUS "${torch_cmake_prefix_path_error}")
	message(FATAL_ERROR "There were problems with the torch configuration")
else ()
	message(STATUS "Found torch API")
	message(STATUS "${torch_cmake_prefix_path_output}")
endif ()


# Torch requires also cuda libraries to be installed
# My in my particular installation I can't run CUDA code, so in principle
# I could install only the cpu torch version, but since other environment
# could have CUDA installed I need to make it a required dependency
list(APPEND CMAKE_PREFIX_PATH ${torch_cmake_prefix_path_output})
find_package(Torch REQUIRED)


# now we can easily compile all the c++ sources into a shared library
# to be loaded at will from the python scripts
set(AGGRESSIVE_COMPILE_OPTIONS "-O3" "-march=native" "-ftree-vectorize" "-ffast-math")
set(CPP_SHARED_LIB cpp_optimized_implementations)
set(CPP_SOURCE_DIR "${CMAKE_SOURCE_DIR}/slmporting/cpp")
file(GLOB CPP_SOURCE_FILES "${CPP_SOURCE_DIR}/*.cpp")

add_library(${CPP_SHARED_LIB} SHARED ${CPP_SOURCE_FILES})
target_compile_options(${CPP_SHARED_LIB} PRIVATE ${AGGRESSIVE_COMPILE_OPTIONS})
target_include_directories(${CPP_SHARED_LIB} PRIVATE ${Python3_INCLUDE_DIRS})
target_include_directories(${CPP_SHARED_LIB} PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(${CPP_SHARED_LIB} PRIVATE ${TORCH_LIBRARIES})
set_target_properties(${CPP_SHARED_LIB} PROPERTIES POSITION_INDEPENDENT_CODE ON)


# Here I want custom rules for invoking the verification and profiling scripts
