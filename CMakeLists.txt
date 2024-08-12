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

## [CMake version]
cmake_minimum_required(VERSION 3.20)

## [Project name and description]
project(CoolDiff VERSION 1.0.0.0
				 DESCRIPTION "CoolDiff - Symbolic/Auto differentiation C++ tool"
				 LANGUAGES C CXX)

## [Custom flags]
# Build type: Debug or Release
set(BUILD_TYPE Release)
option(BUILD_TEST "Building unit test executables" TRUE)
option(BUILD_EXAMPLE "Building example executables" TRUE)

## [CMAKE internal flags]
# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type
if(${BUILD_TYPE} STREQUAL Debug)
	set(CMAKE_BUILD_TYPE Debug)
	add_definitions(-DBUILD_TYPE)
	message("[${BUILD_TYPE}] Generating debug code")
else()
	set(CMAKE_BUILD_TYPE Release)
	message("[${BUILD_TYPE}] Generating production code")
endif()

#LTO
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

## [External dependencies]
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

## [Set compilation flags]
if (${BUILD_TYPE} STREQUAL Debug)
	add_definitions(-DSCALAR_TYPE=double)
	add_definitions(-DUSE_COMPLEX_MATH)
	add_definitions(-DUSE_ROBIN_HOOD_MAP)
	add_definitions(-DUSE_VIRTUAL_FUNCTIONS)
endif()

# [Set target sources]
set(SOURCES ${CMAKE_SOURCE_DIR}/src/CommonHeader.cpp
			${CMAKE_SOURCE_DIR}/src/VarWrap.cpp
			${CMAKE_SOURCE_DIR}/src/Variable.cpp
			${CMAKE_SOURCE_DIR}/src/MetaVariable.cpp
			${CMAKE_SOURCE_DIR}/src/Expression.cpp
			${CMAKE_SOURCE_DIR}/src/CommonFunctions.cpp
			${CMAKE_SOURCE_DIR}/src/MemoryManager.cpp
			${CMAKE_SOURCE_DIR}/src/Parameter.cpp)

# [Set target includes]
set(INCLUDES ${CMAKE_SOURCE_DIR}/include
			 ${CMAKE_SOURCE_DIR}/include/BinaryOps
			 ${CMAKE_SOURCE_DIR}/include/UnaryOps
			 ${CMAKE_BINARY_DIR}/_deps/robin-hood-hashing-src/src/include)

# [Add executable (target executable)]
add_library(${PROJECT_NAME} STATIC ${SOURCES})
# [Set target include directories]
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDES})	
# [Set compile options]
target_compile_options(${PROJECT_NAME} PRIVATE -fpermissive)

if(${BUILD_TYPE} STREQUAL Debug)
	message("[${BUILD_TYPE}] Attaching address sanitizer")
	target_compile_options(${PROJECT_NAME} PRIVATE -fno-omit-frame-pointer -fsanitize=address)
	target_link_options(${PROJECT_NAME} PRIVATE -fno-omit-frame-pointer -fsanitize=address)
else()
	message("[${BUILD_TYPE}] Address sanitizer disabled")
endif()

if(${BUILD_TEST})
	include(CTest)
	enable_testing()
	add_subdirectory(${CMAKE_SOURCE_DIR}/test)
else()
	message("[${BUILD_TYPE}] Unit test build is skipped")
endif()

if(${BUILD_EXAMPLE})
	add_subdirectory(${CMAKE_SOURCE_DIR}/example)
else()
	message("[${BUILD_TYPE}] Example build is skipped")
endif()
