# set required cmake version
cmake_minimum_required(VERSION 3.19...3.30)

project(
  mqt-mlir
  LANGUAGES C CXX
  DESCRIPTION "MQT MLIR - Interfacing between the Munich Quantum Toolkit and MLIR")

# Add path for custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# check if this is the master project or used via add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  set(MQT_MLIR_MASTER_PROJECT ON)
else()
  set(MQT_MLIR_MASTER_PROJECT OFF)
endif()

option(BUILD_MQT_MLIR_TESTS "Also build tests for the MQT MLIR project" ${MQT_MLIR_MASTER_PROJECT})

# Require C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_C_EXTENSIONS False)

# Require C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(cmake/ExternalDependencies.cmake)

# set the include directory for the build tree
set(MQT_MLIR_INCLUDE_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# add main library code
add_subdirectory(src)

# add test code
if(BUILD_MQT_MLIR_TESTS)
  enable_testing()
  include(GoogleTest)
  add_subdirectory(test)
endif()
