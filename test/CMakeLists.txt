#[[
/**
* @file test/CMakeLists.txt
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

include(GoogleTest)

set(TEST_EXECUTABLE OpsTest)
if(${BUILD_TEST})
    # Add executable (target testing)
    add_executable(${TEST_EXECUTABLE} ${CMAKE_SOURCE_DIR}/test/OpsTest.cpp)
    # Set target include directories
    target_include_directories(${TEST_EXECUTABLE} PUBLIC ${INCLUDES})
    # Set compile options
    target_compile_options(${TEST_EXECUTABLE} PRIVATE -fpermissive)
    if(${BUILD_TYPE} STREQUAL Debug)
        target_compile_options(
            ${TEST_EXECUTABLE} PRIVATE -fno-omit-frame-pointer
                                       -fsanitize=address)
        target_link_options(
            ${TEST_EXECUTABLE}
            PRIVATE
            -fno-omit-frame-pointer
            -fsanitize=address)
    endif()
    target_link_libraries(${TEST_EXECUTABLE} PRIVATE gtest_main CoolDiff)
endif()

gtest_discover_tests(${TEST_EXECUTABLE})
