cmake_minimum_required(VERSION 3.20)
project(slm-porting CXX)

find_package(Python3 REQUIRED)

# for now the python tests are made by calling independent programs that perform unit testing
enable_testing()
add_test(NAME "rng"   COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/python/rng.py")
add_test(NAME "slm"   COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/python/slm.py")
add_test(NAME "units" COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/python/units.py")
