## [CMake version]
cmake_minimum_required(VERSION 3.20)

## [Project name and description]
project(CoolDiff VERSION 1.0.0.0
				 DESCRIPTION "CoolDiff - Symbolic/Auto differentiation C++ tool"
				 LANGUAGES C CXX)

## [Custom flags]
# Build type: Debug or Release
set(BUILD_TYPE Release)
set(BUILD_TEST TRUE)
set(BUILD_EXAMPLE TRUE)

## [CMAKE internal flags]
# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type
if(${BUILD_TYPE} STREQUAL Debug)
	set(CMAKE_BUILD_TYPE Debug)
	add_definitions(-DBUILD_TYPE)
	message("[Debug code] Generating debug code")
else()
	set(CMAKE_BUILD_TYPE Release)
	message("[Production code] Generating production code")
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

FetchContent_Declare(
	googletest
	GIT_REPOSITORY https://github.com/google/googletest.git
	GIT_TAG v1.15.2
	GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(robin-hood-hashing)
FetchContent_MakeAvailable(googletest)

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
	target_compile_options(${PROJECT_NAME} PRIVATE -fsanitize=address)
	target_link_options(${PROJECT_NAME} PRIVATE -fsanitize=address)
endif()

if(${BUILD_TEST})
	add_subdirectory(${CMAKE_SOURCE_DIR}/test)
endif()

if(${BUILD_EXAMPLE})
	add_subdirectory(${CMAKE_SOURCE_DIR}/example)
endif()