#[[
/**
* @file CMakeLists.txt
*
* @copyright 2023-2025 Karthik Murali Madhavan Rathai
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

# TODO - Enable shared library build 
option(BUILD_UNITTESTS "Building unit test executables" ON) # Default - ON
option(BUILD_EXAMPLES "Building example executables" OFF) # Default - ON
option(ENABLE_CLANG_TIDY "Enable clang-tidy static linter" OFF) # Default - OFF
option(ENABLE_CMAKE_FORMAT "Enable cmake formatter" OFF) # Default - OFF
option(ENABLE_CLANG_FORMAT "Enable clang format" OFF) # Default - OFF
option(ENABLE_COMPILER_WARNINGS "Enable compiler warnings" OFF) # Default - OFF
option(ENABLE_COMPLEX_MATH "Enable complex numbers for numerics" ON) # Default - ON
option(ENABLE_ROBIN_HOOD_MAP "Enable robin hood hashmaps" OFF) # Default - ON
# TODO - Implementation of manual dynamic dispatch for performance
option(ENABLE_VIRTUAL_FUNCTIONS "Enable virtual functions" ON) # Default - ON
option(ENABLE_DYNAMIC_ASSERTIONS "Enable dynamic assertions" ON) # Default - OFF
option(ENABLE_STL_PARALLEL_EXECUTION "Enable STL libraries parallel execution feature" OFF) # Default - OFF
set(DEFAULT_MATH_TYPE 2 CACHE STRING "Value that defined the default math type: Float  - 1, Double - 2") # Default - 2 for test cases


########################################################################################################################
########################################################################################################################
########################################################################################################################
                                    # [External dependencies]
########################################################################################################################
########################################################################################################################
######################################################################################################################## 

include(FetchContent)

# Find and install robin hood hashing package
if(${ENABLE_ROBIN_HOOD_MAP})
    find_package(robin_hood QUIET)
    if(DEFINED robin_hood_FOUND AND DEFINED robin_hood_VERSION)
        message("[${CMAKE_BUILD_TYPE}][${PROJECT_NAME}] Robin hood installed version ${robin_hood_VERSION}")
    endif()

    if(NOT robin_hood_FOUND)
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
    else()
        message("[${CMAKE_BUILD_TYPE}][${PROJECT_NAME}] Robin hood library found in system")
    endif()
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
        # Memory manager folder
        ${CMAKE_SOURCE_DIR}/src/MemoryManager/MemoryManager.cpp)

########################################################################################################################
########################################################################################################################
########################################################################################################################
                                # [Set target includes]
########################################################################################################################
########################################################################################################################
########################################################################################################################

set(PRIVATE_INCLUDES    # Matrix handler include directory
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler

                        # Naive CPU addition handlers
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixAddition/NaiveCPU
                        # Eigen library addition handler
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixAddition/Eigen

                        # Naive CPU multiplication handlers
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixMultiplication/NaiveCPU
                        # Eigen library multiplication handler
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixMultiplication/Eigen

                        # Naive CPU subtraction handlers
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixSubtraction/NaiveCPU
                        # Eigen library subtraction handler
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixSubtraction/Eigen

                        # Naive CPU Hadamard handlers
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixHadamard/NaiveCPU
                        # Eigen library Hadamard handler
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixHadamard/Eigen

                        # Naive CPU Kronecker product handlers
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixKronProduct/NaiveCPU
                        # Eigen library Kronecker handler
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixKronProduct/Eigen

                        # Naive CPU Transpose handlers
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixTranspose/NaiveCPU
                        # Eigen Transpose handlers
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixTranspose/Eigen

                        # Eigen determinant handlers
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixDet/Eigen

                        # Naive trace handlers
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixTrace/NaiveCPU
                        # Eigen trace handlers
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixTrace/Eigen
                        
                        # Naive unary handlers
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixUnary/NaiveCPU

                        # Eigen inverse handlers
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixInverse/Eigen

                        # Naive Convolution handlers            
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixConvolution/NaiveCPU
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixHandler/MatrixConvolution/Eigen)

set(PUBLIC_INCLUDES     #General include (CommonHeader.hpp, CoolDiff.hpp)
                        ${CMAKE_SOURCE_DIR}/include

                        # Scalar classes (Variable, Parameter, Expression), unary and binary ops
                        ${CMAKE_SOURCE_DIR}/include/Scalar
                        ${CMAKE_SOURCE_DIR}/include/Scalar/BinaryOps
                        ${CMAKE_SOURCE_DIR}/include/Scalar/UnaryOps
                        
                        # Matrix class 
                        ${CMAKE_SOURCE_DIR}/include/Matrix
                        ${CMAKE_SOURCE_DIR}/include/Matrix/MatrixInterface

                        # Matrix unary and binary ops
                        ${CMAKE_SOURCE_DIR}/include/Matrix/BinaryOps
                        ${CMAKE_SOURCE_DIR}/include/Matrix/UnaryOps
                        
                        # Memory manager
                        ${CMAKE_SOURCE_DIR}/include/MemoryManager)

########################################################################################################################
########################################################################################################################
########################################################################################################################
                                # [Set target properties]
########################################################################################################################
########################################################################################################################
########################################################################################################################

# [Add executable (target executable)]
add_library(${PROJECT_NAME})
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

set(LIBRARY_LIST Threads::Threads Eigen3::Eigen)
if(${ENABLE_ROBIN_HOOD_MAP})
    list(APPEND LIBRARY_LIST robin_hood::robin_hood)
endif()

# [Set target sources]
target_sources(${PROJECT_NAME} PRIVATE ${SRC})

# [Set target include directories]
target_include_directories(${PROJECT_NAME}  PRIVATE ${PRIVATE_INCLUDES}
                                            PUBLIC  $<BUILD_INTERFACE:;${PUBLIC_INCLUDES}> 
                                                    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>)

# [Set target properties]
set_target_properties(${PROJECT_NAME}   PROPERTIES  VERSION ${PROJECT_VERSION}
                                                    SOVERSION ${PROJECT_VERSION_MAJOR})

# [Link target libraries]
target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBRARY_LIST})

# Attach address sanitizer
set(HOOK_ADDRESS_SANITIZER  $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang>>:-fno-omit-frame-pointer;-fsanitize=address>
                            $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:MSVC>>:/INCREMENTAL:NO;/fsanitize=address>)
# Set permission flags
set(COMPILE_PERMISSION_FLAGS $<$<CXX_COMPILER_ID:GNU,Clang>:-fpermissive> 
                    $<$<CXX_COMPILER_ID:MSVC>:/permissive>)
# Set warning flags
set(COMPILE_WARNING_FLAGS $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:GNU,Clang>>:-Wpedantic;-Wall;-Wextra>
                $<$<AND:$<CONFIG:Debug>,$<CXX_COMPILER_ID:MSVC>>:/W4>)   

# [Set compile options]
if(${ENABLE_COMPILER_WARNINGS})
    target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_WARNING_FLAGS})
endif()
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_PERMISSION_FLAGS})
target_compile_options(${PROJECT_NAME} PRIVATE ${HOOK_ADDRESS_SANITIZER})

# [Set link options]
target_link_options(${PROJECT_NAME} PRIVATE ${HOOK_ADDRESS_SANITIZER})

# [Set compilation flags]
if(${DEFAULT_MATH_TYPE} EQUAL 1)
    target_compile_definitions(${PROJECT_NAME} PUBLIC COOLDIFF_SCALAR_TYPE=1) # Float
elseif(${DEFAULT_MATH_TYPE} EQUAL 2)
    target_compile_definitions(${PROJECT_NAME} PUBLIC COOLDIFF_SCALAR_TYPE=2) # Double
endif()

if(${ENABLE_DYNAMIC_ASSERTIONS})
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_DYNAMIC_ASSERTIONS)
endif()

if(${ENABLE_COMPLEX_MATH})
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_COMPLEX_MATH)
endif()

if(${ENABLE_ROBIN_HOOD_MAP})
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_ROBIN_HOOD_MAP)
endif()

if(${ENABLE_VIRTUAL_FUNCTIONS})
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_VIRTUAL_FUNCTIONS)
endif()

if(${ENABLE_STL_PARALLEL_EXECUTION})
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_CXX_PARALLEL_POLICY)
endif() 

########################################################################################################################
########################################################################################################################
########################################################################################################################
                                # [Install section]
########################################################################################################################
########################################################################################################################
########################################################################################################################

if(NOT MSVC)
    include(GNUInstallDirs)

    install(TARGETS ${PROJECT_NAME}
            EXPORT "${PROJECT_NAME}Targets"
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

    install(DIRECTORY ${CMAKE_SOURCE_DIR}/include 
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

    include(CMakePackageConfigHelpers)
    configure_package_config_file(  "${CMAKE_SOURCE_DIR}/config/${PROJECT_NAME}Config.cmake.in" 
                                    "${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake"    
                                    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" )

    write_basic_package_version_file(   "${PROJECT_NAME}ConfigVersion.cmake"
                                        VERSION ${PROJECT_VERSION}
                                        COMPATIBILITY SameMajorVersion  )

    install(FILES   "${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake" 
                    "${CMAKE_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" )

    install(    EXPORT "${PROJECT_NAME}Targets"
                FILE "${PROJECT_NAME}Targets.cmake"
                NAMESPACE "${PROJECT_NAME}::"
                DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" )
endif()

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