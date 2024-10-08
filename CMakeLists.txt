#[[
/**
* @file CMakeLists.txt
*
* @copyright 2023-2024 Karthik Murali Madhavan Rathai
*/
/*
* This file is part of CoolDiff library.
*
* You can redistribute it and/or modify it under the terms of the GNU
* General Public License version 3 as published by the Free Software
* Foundation.
*
* Licensees holding a valid commercial license may use this software
* in accordance with the commercial license agreement provided in
* conjunction with the software.  The terms and conditions of any such
* commercial license agreement shall govern, supersede, and render
* ineffective any application of the GPLv3 license to this software,
* notwithstanding of any reference thereto in the software or
* associated repository.
*/
]]

# [CMake version]
cmake_minimum_required(VERSION 3.20)

# [Project name and description]
project(
    CoolDiff
    VERSION 1.0.1.0
    DESCRIPTION "CoolDiff - Symbolic/Auto differentiation C++ tool"
    LANGUAGES C CXX)

find_package(TBB REQUIRED)
find_package(Threads REQUIRED)

# [Custom flags]
# Build type: Debug or Release
set(BUILD_TYPE Debug)

option(BUILD_TEST "Building unit test executables" ON)
option(BUILD_EXAMPLE "Building example executables" ON)
option(ENABLE_CLANG_TIDY "Enable clang-tidy static linter" ON)
option(ENABLE_CMAKE_FORMAT "Enable cmake formatter" ON)
option(ENABLE_CLANG_FORMAT "Enable clang format" ON)

# [CMAKE internal flags]
# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Include CMake scripts
include(Tools)

# Set build type
if(${BUILD_TYPE} STREQUAL Debug)
    set(CMAKE_BUILD_TYPE Debug)
    add_definitions(-DBUILD_TYPE)
    message("[${BUILD_TYPE}][${PROJECT_NAME}] Generating debug code")
else()
    set(CMAKE_BUILD_TYPE Release)
    message("[${BUILD_TYPE}][${PROJECT_NAME}] Generating production code")
endif()

#LTO
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# [External dependencies]
# Find google test package
include(FetchContent)
FetchContent_Declare(
    robin-hood-hashing
    GIT_REPOSITORY https://github.com/martinus/robin-hood-hashing.git
    GIT_TAG 3.11.5
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(robin-hood-hashing)

if(${BUILD_TEST})
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
        GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(googletest)
    list(APPEND CMAKE_MODULE_PATH ${gtest_SOURCE_DIR}/cmake)
endif()

# [Set compilation flags]
if(${BUILD_TYPE} STREQUAL Debug)
    add_definitions(-DSCALAR_TYPE=double)
    add_definitions(-DUSE_COMPLEX_MATH)
    add_definitions(-DUSE_ROBIN_HOOD_MAP)
    add_definitions(-DUSE_VIRTUAL_FUNCTIONS)
    add_definitions(-DUSE_PARALLEL_POLICY)
endif()

# [Set target sources]
set(SRC
    ${CMAKE_SOURCE_DIR}/src/CommonHeader.cpp
    ${CMAKE_SOURCE_DIR}/src/Scalar/VarWrap.cpp
    ${CMAKE_SOURCE_DIR}/src/Scalar/Variable.cpp
    ${CMAKE_SOURCE_DIR}/src/Scalar/Parameter.cpp
    ${CMAKE_SOURCE_DIR}/src/Scalar/MetaVariable.cpp
    ${CMAKE_SOURCE_DIR}/src/Scalar/Expression.cpp
    ${CMAKE_SOURCE_DIR}/src/Scalar/CommonFunctions.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/CommonMatFunctions.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatOperators.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixBasics.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/Matrix.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixSplOps/MatrixZeroOps.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixSplOps/MatrixEyeOps.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixHandler.cpp
    # MatrixAddHandler folder
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixAddHandler/ZeroMatAddHandler.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixAddHandler/ZeroMatScalarAddHandler.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixAddHandler/EyeMatAddHandler.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixAddHandler/EyeMatScalarAddHandler.cpp
    # MatrixAddition folder
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixAddition/MatAddNaiveHandler.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixAddition/MatScalarAddNaiveHandler.cpp
    # MatrixKronHandler folder
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixKronHandler/ZeroMatKronHandler.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixKronHandler/EyeMatKronHandler.cpp
    # MatrixKronProduct folder
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixKronProduct/MatKronNaiveHandler.cpp
    # MatrixMulHandler folder
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixMulHandler/ZeroMatMulHandler.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixMulHandler/ZeroMatScalarMulHandler.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixMulHandler/EyeMatMulHandler.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixMulHandler/EyeMatScalarMulHandler.cpp
    # MatrixMultiplication folder
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixMultiplication/MatMulNaiveHandler.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixMultiplication/MatScalarMulNaiveHandler.cpp
    # MatrixSubHandler folder
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixSubHandler/ZeroMatSubHandler.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixSubHandler/EyeMatSubHandler.cpp
    # MatrixSubtraction folder
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixSubtraction/MatSubNaiveHandler.cpp
    # MatrixHadamardHandler folder
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixHadamardHandler/ZeroMatHadamardHandler.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixHadamardHandler/EyeMatHadamardHandler.cpp
    # MatrixHadamard folder
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixHadamard/MatHadamardNaiveHandler.cpp
    # MatrixTransposeHandler folder
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixTransposeHandler/ZeroMatTransposeHandler.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixTransposeHandler/EyeMatTransposeHandler.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixTransposeHandler/ZeroMatDervTransposeHandler.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixTransposeHandler/EyeMatDervTransposeHandler.cpp
    # MatrixHadamard folder
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixTranspose/MatTransposeNaiveHandler.cpp
    ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixHandler/MatrixTranspose/MatDervTransposeNaiveHandler.cpp
    # Memory manager folder
    ${CMAKE_SOURCE_DIR}/src/MemoryManager/MemoryManager.cpp)

# [Set target includes]
set(INCLUDES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/Scalar/BinaryOps
    ${CMAKE_SOURCE_DIR}/include/Scalar/UnaryOps
    ${CMAKE_SOURCE_DIR}/include/Scalar
    ${CMAKE_SOURCE_DIR}/include/Matrix
    ${CMAKE_SOURCE_DIR}/include/Matrix/BinaryOps
    ${CMAKE_SOURCE_DIR}/include/Matrix/UnaryOps
    ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler
    ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixAddHandler
    ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixAddition
    ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixKronHandler
    ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixKronProduct
    ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixMulHandler
    ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixMultiplication
    ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixSubHandler 
    ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixSubtraction
    ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixHadamardHandler 
    ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixHadamard
    ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixTransposeHandler
    ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixTranspose
    ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixSplOps
    ${CMAKE_SOURCE_DIR}/include/MemoryManager
    ${CMAKE_BINARY_DIR}/_deps/robin-hood-hashing-src/src/include)

# [Add executable (target executable)]
add_library(${PROJECT_NAME} STATIC ${SRC})
# [Set target include directories]
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDES})
# [Set compile options]
target_compile_options(${PROJECT_NAME} PRIVATE -fpermissive)
target_link_libraries(${PROJECT_NAME} PUBLIC Threads::Threads TBB::tbb)

if(${BUILD_TYPE} STREQUAL Debug)
    message("[${BUILD_TYPE}][${PROJECT_NAME}] Attaching address sanitizer")
    target_compile_options(${PROJECT_NAME} PRIVATE -fno-omit-frame-pointer
                                                   -fsanitize=address)
    target_link_options(
        ${PROJECT_NAME}
        PRIVATE
        -fno-omit-frame-pointer
        -fsanitize=address)
else()
    message("[${BUILD_TYPE}][${PROJECT_NAME}] Address sanitizer disabled")
endif()

#[[--------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
-----------------------------------------------------------------------------
]]

# Build test scripts
if(${BUILD_TEST})
    include(CTest)
    enable_testing()
    add_subdirectory(${CMAKE_SOURCE_DIR}/test)
else()
    message("[${BUILD_TYPE}][${PROJECT_NAME}] Unit test build is skipped")
endif()

# Build examples
if(${BUILD_EXAMPLE})
    add_subdirectory(${CMAKE_SOURCE_DIR}/example)
else()
    message("[${BUILD_TYPE}][${PROJECT_NAME}] Example build is skipped")
endif()

# Enable clang tidy
if(${ENABLE_CLANG_TIDY})
    add_clang_tidy_to_target(${PROJECT_NAME})
else()
    message("[${BUILD_TYPE}][${PROJECT_NAME}] Clang-tidy disabled")
endif()

# Enable cmake formatter
if(${ENABLE_CMAKE_FORMAT})
    add_cmake_format_target()
else()
    message("[${BUILD_TYPE}][${PROJECT_NAME}] CMake formatter disabled")
endif()

# Enable CLANG formatter
if(${ENABLE_CLANG_FORMAT})
    add_clang_format_target()
else()
    message("[${BUILD_TYPE}][${PROJECT_NAME}] Clang formatter disabled")
endif()
