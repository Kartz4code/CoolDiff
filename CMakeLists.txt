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
cmake_minimum_required(VERSION 3.22)

# [Project name and description]
project(CoolDiff    VERSION 1.0.1.0
                    DESCRIPTION "CoolDiff - A Symbolic/Automatic differentiation C++ tool"
                    LANGUAGES C CXX)

########################################################################################################################
########################################################################################################################
########################################################################################################################
                                    # [CMAKE internal flags]
########################################################################################################################
########################################################################################################################
######################################################################################################################## 

# Custom flags
set(CMAKE_BUILD_TYPE Debug)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#LTO and IPO
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set CXX flags for maximum performance
if(${CMAKE_BUILD_TYPE} EQUAL Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -ffast-math -funroll-loops -funroll-all-loops -ftree-vectorize -finline-functions -floop-block")
endif()

# Include CMake scripts
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(Tools)

########################################################################################################################
########################################################################################################################
########################################################################################################################
                                    # [CMAKE build options]
########################################################################################################################
########################################################################################################################
######################################################################################################################## 

option(BUILD_UNITTESTS "Building unit test executables" ON)
option(BUILD_EXAMPLES "Building example executables" ON)
option(ENABLE_CLANG_TIDY "Enable clang-tidy static linter" OFF)
option(ENABLE_CMAKE_FORMAT "Enable cmake formatter" OFF)
option(ENABLE_CLANG_FORMAT "Enable clang format" OFF)
option(ENABLE_COMPILER_WARNINGS "Enable compiler warnings" OFF)
option(ENABLE_COMPLEX_MATH "Enable complex numbers for numerics" ON)
option(ENABLE_ROBIN_HOOD_MAP "Enable robin hood hashmaps" ON)
# TODO - Implementation of manual dynamic dispatch for performance
option(ENABLE_VIRTUAL_FUNCTIONS "Enable virtual functions" ON)
option(ENABLE_DYNAMIC_ASSERTIONS "Enable dynamic assertions" ON)
option(ENABLE_STL_PARALLEL_EXECUTION "Enable STL libraries parallel execution feature" OFF)
option(ENABLE_SYMBOLIC_CHECK "Enable symbolic checks" ON)
# TODO - Verification of numerical checks
option(ENABLE_NUMERICAL_CHECK "Enable numerical checks" OFF)
set(DEFAULT_MATH_TYPE 2 CACHE STRING "Value that defined the default math type: Float  - 1, 
                                                                                Double - 2")

# [Set compilation flags]
if(${DEFAULT_MATH_TYPE} EQUAL 1)
    add_definitions(-DCOOLDIFF_SCALAR_TYPE=1) # Float
elseif(${DEFAULT_MATH_TYPE} EQUAL 2)
    add_definitions(-DCOOLDIFF_SCALAR_TYPE=2) # Double
endif()

if(${ENABLE_DYNAMIC_ASSERTIONS})
    add_definitions(-DUSE_DYNAMIC_ASSERTIONS)
endif()

if(${ENABLE_COMPLEX_MATH})
    add_definitions(-DUSE_COMPLEX_MATH)
endif()

if(${ENABLE_ROBIN_HOOD_MAP})
    add_definitions(-DUSE_ROBIN_HOOD_MAP)
endif()

if(${ENABLE_VIRTUAL_FUNCTIONS})
    add_definitions(-DUSE_VIRTUAL_FUNCTIONS)
endif()

if(${ENABLE_STL_PARALLEL_EXECUTION})
    add_definitions(-DUSE_CXX_PARALLEL_POLICY)
endif()

if(${ENABLE_SYMBOLIC_CHECK})
    add_definitions(-DUSE_SYMBOLIC_CHECK)
endif()

if(${ENABLE_NUMERICAL_CHECK})
    add_definitions(-DUSE_NUMERICAL_CHECK)
endif()    


########################################################################################################################
########################################################################################################################
########################################################################################################################
                                    # [External dependencies]
########################################################################################################################
########################################################################################################################
######################################################################################################################## 

include(FetchContent)

# Find and install robin hood hashing package
FetchContent_Declare(
    robin-hood-hashing
    GIT_REPOSITORY https://github.com/martinus/robin-hood-hashing.git
    GIT_TAG 3.11.5
    GIT_SHALLOW TRUE
) 
# Check if the dependency has already been downloaded and built
FetchContent_GetProperties(robin-hood-hashing)
if(NOT robin-hood-hashing_POPULATED)
    message("[${CMAKE_BUILD_TYPE}][${PROJECT_NAME}] Setting robin-hood library")
    FetchContent_MakeAvailable(robin-hood-hashing)
endif()

# Find and install Eigen package
find_package(Eigen3 QUIET)
if(DEFINED Eigen3_FOUND AND DEFINED Eigen3_VERSION)
    message("[${CMAKE_BUILD_TYPE}][${PROJECT_NAME}] Eigen installed version ${Eigen3_VERSION}")
endif()
if(NOT Eigen3_FOUND OR Eigen3_VERSION VERSION_LESS "3.4")
    FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG master
    GIT_SHALLOW TRUE
    ) 
    # Check if the dependency has already been downloaded and built
    FetchContent_GetProperties(Eigen)
    if(NOT Eigen_POPULATED)
        message("[${CMAKE_BUILD_TYPE}][${PROJECT_NAME}] Setting latest Eigen library")
        FetchContent_MakeAvailable(Eigen)
    endif()
else()
    message("[${CMAKE_BUILD_TYPE}][${PROJECT_NAME}] Eigen library found in system")
endif()

# Find and install Gtest package for test build
find_package(GTest QUIET)
if(DEFINED GTest_FOUND AND DEFINED GTest_VERSION)
    message("[${CMAKE_BUILD_TYPE}][${PROJECT_NAME}] Gtest installed version ${GTest_VERSION}")
endif()
if(NOT GTest_FOUND OR GTest_VERSION VERSION_LESS "1.15")
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
        GIT_SHALLOW TRUE
    )
    # Check if the dependency has already been downloaded and built
    FetchContent_GetProperties(googletest)
    if(NOT googletest_POPULATED)
        message("[${CMAKE_BUILD_TYPE}][${PROJECT_NAME}] Setting latest Gtest library")
        FetchContent_MakeAvailable(googletest)
    endif()
else()
    message("[${CMAKE_BUILD_TYPE}][${PROJECT_NAME}] Gtest library found in system")
endif()

# Enable Threads package
find_package(Threads REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${gtest_SOURCE_DIR}/cmake)

########################################################################################################################
########################################################################################################################
########################################################################################################################
                                # [Set target sources]
########################################################################################################################
########################################################################################################################
########################################################################################################################

set(SRC # Scalar folder *.cpp files
        ${CMAKE_SOURCE_DIR}/src/CommonHeader.cpp
        ${CMAKE_SOURCE_DIR}/src/Scalar/VarWrap.cpp
        ${CMAKE_SOURCE_DIR}/src/Scalar/Variable.cpp
        ${CMAKE_SOURCE_DIR}/src/Scalar/Parameter.cpp
        ${CMAKE_SOURCE_DIR}/src/Scalar/MetaVariable.cpp
        ${CMAKE_SOURCE_DIR}/src/Scalar/Expression.cpp
        ${CMAKE_SOURCE_DIR}/src/Scalar/CommonFunctions.cpp
        # Matrix folder *.cpp files
        ${CMAKE_SOURCE_DIR}/src/Matrix/CommonMatFunctions.cpp
        ${CMAKE_SOURCE_DIR}/src/Matrix/MatOperators.cpp
        ${CMAKE_SOURCE_DIR}/src/Matrix/MetaMatrix.cpp
        ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixBasics.cpp
        ${CMAKE_SOURCE_DIR}/src/Matrix/Matrix.cpp
        ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixSplOps/MatrixZeroOps.cpp
        ${CMAKE_SOURCE_DIR}/src/Matrix/MatrixSplOps/MatrixEyeOps.cpp
        # Memory manager folder
        ${CMAKE_SOURCE_DIR}/src/MemoryManager/MemoryManager.cpp)

########################################################################################################################
########################################################################################################################
########################################################################################################################
                                # [Set target includes]
########################################################################################################################
########################################################################################################################
########################################################################################################################

set(INCLUDES    ${CMAKE_SOURCE_DIR}/include
                ${CMAKE_SOURCE_DIR}/include/Scalar/BinaryOps
                ${CMAKE_SOURCE_DIR}/include/Scalar/UnaryOps
                ${CMAKE_SOURCE_DIR}/include/Scalar
                ${CMAKE_SOURCE_DIR}/include/Matrix
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixInterface
                ${CMAKE_SOURCE_DIR}/include/Matrix/BinaryOps
                ${CMAKE_SOURCE_DIR}/include/Matrix/UnaryOps
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler
            
                # Naive CPU addition handlers
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixAddHandler/NaiveCPU
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixAddition/NaiveCPU
                
                # Eigen library addition handler
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixAddition/Eigen
                
                # Naive CPU multiplication handlers
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixMulHandler/NaiveCPU
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixMultiplication/NaiveCPU

                # Eigen library multiplication handler
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixMultiplication/Eigen

                # Naive CPU subtraction handlers
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixSubHandler/NaiveCPU
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixSubtraction/NaiveCPU

                # Eigen library subtraction handler
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixSubtraction/Eigen

                # Naive CPU Hadamard handlers
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixHadamardHandler/NaiveCPU
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixHadamard/NaiveCPU

                # Eigen library Hadamard handler
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixHadamard/Eigen

                # Naive CPU Kronecker product handlers
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixKronHandler/NaiveCPU
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixKronProduct/NaiveCPU

                # Eigen library Kronecker handler
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixKronProduct/Eigen

                # Naive CPU Transpose product handlers
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixTransposeHandler/NaiveCPU
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixTranspose/NaiveCPU

                # Determinant handlers
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixDet/Eigen
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixDetHandler/NaiveCPU

                # Unary handlers
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixUnaryHandler/NaiveCPU
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixUnary/NaiveCPU

                # Inverse handlers
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixInverse/Eigen
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixInvHandler/NaiveCPU

                # Convolution handlers            
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixConvHandler/NaiveCPU
                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixConvolution/NaiveCPU

                ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixSplOps
                ${CMAKE_SOURCE_DIR}/include/MemoryManager
                ${CMAKE_BINARY_DIR}/_deps/robin-hood-hashing-src/src/include)

########################################################################################################################
########################################################################################################################
########################################################################################################################
                                # [Set target properties]
########################################################################################################################
########################################################################################################################
########################################################################################################################

# [Add executable (target executable)]
add_library(${PROJECT_NAME} STATIC ${SRC})

# [Set target include directories]
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDES})

# [Link target libraries]
target_link_libraries(${PROJECT_NAME} PUBLIC Threads::Threads Eigen3::Eigen)

# Attach address sanitizer
set(ATTACH_ADDRESS_SANITIZER $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang>>:-fno-omit-frame-pointer;-fsanitize=address>
                            $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:MSVC>>:/INCREMENTAL:NO;/fsanitize=address>)
# Set permission flags
set(PERMISSION_FLAGS $<$<CXX_COMPILER_ID:GNU,Clang>:-fpermissive> 
                    $<$<CXX_COMPILER_ID:MSVC>:/permissive>)
# Set warning flags
set(WARNING_FLAGS $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wpedantic;-Wall;-Wextra>
                $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:MSVC>>:/W4>)   

# [Set compile options]
if(${ENABLE_COMPILER_WARNINGS})
    target_compile_options(${PROJECT_NAME} PRIVATE ${WARNING_FLAGS})
endif()
target_compile_options(${PROJECT_NAME} PRIVATE ${PERMISSION_FLAGS})
target_compile_options(${PROJECT_NAME} PRIVATE ${ATTACH_ADDRESS_SANITIZER})

# [Set link options]
target_link_options(${PROJECT_NAME} PRIVATE ${ATTACH_ADDRESS_SANITIZER})



########################################################################################################################
########################################################################################################################
########################################################################################################################
                                    # 1. [Build test scripts]
                                    # 2. [Build examples]
                                    # 3. [Enable clang tidy]
                                    # 4. [Enable cmake formatter]
                                    # 5. [Enable clang formatter]
########################################################################################################################
########################################################################################################################
######################################################################################################################## 

# Build test scripts
if(${BUILD_UNITTESTS})
    include(CTest)
    enable_testing()
    add_subdirectory(${CMAKE_SOURCE_DIR}/test)
else()
    message("[${CMAKE_BUILD_TYPE}][${PROJECT_NAME}] Unit test build is skipped")
endif()

# Build examples
if(${BUILD_EXAMPLES})
    add_subdirectory(${CMAKE_SOURCE_DIR}/example)
else()
    message("[${CMAKE_BUILD_TYPE}][${PROJECT_NAME}] Example build is skipped")
endif()

# Enable clang tidy
if(${ENABLE_CLANG_TIDY})
    add_clang_tidy_to_target(${PROJECT_NAME})
else()
    message("[${CMAKE_BUILD_TYPE}][${PROJECT_NAME}] Clang-tidy disabled")
endif()

# Enable cmake formatter
if(${ENABLE_CMAKE_FORMAT})
    add_cmake_format_target()
else()
    message("[${CMAKE_BUILD_TYPE}][${PROJECT_NAME}] CMake formatter disabled")
endif()

# Enable clang formatter
if(${ENABLE_CLANG_FORMAT})
    add_clang_format_target()
else()
    message("[${CMAKE_BUILD_TYPE}][${PROJECT_NAME}] Clang formatter disabled")
endif()