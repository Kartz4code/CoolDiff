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
set(TEST_EXECUTABLE2 MatTest)

if(${BUILD_UNITTESTS} AND NOT MSVC)
##################################################################################################################################
    
    # [Add executable (target testing)]
    add_executable(${TEST_EXECUTABLE} ${CMAKE_SOURCE_DIR}/test/OpsTest.cpp)

    # [Set target include directories]
    target_include_directories(${TEST_EXECUTABLE} PUBLIC ${PUBLIC_INCLUDES})

    # [Link target libraries]
    target_link_libraries(${TEST_EXECUTABLE} PRIVATE GTest::gtest GTest::gtest_main CoolDiff)

    # [Set compile options]
    if(${ENABLE_COMPILER_WARNINGS})
        target_compile_options(${TEST_EXECUTABLE} PRIVATE ${COMPILE_WARNING_FLAGS})
    endif()
    target_compile_options(${TEST_EXECUTABLE} PRIVATE ${COMPILE_PERMISSION_FLAGS})
    target_compile_options(${TEST_EXECUTABLE} PRIVATE ${HOOK_ADDRESS_SANITIZER})

    # [Set link options]
    target_link_options(${TEST_EXECUTABLE} PRIVATE ${HOOK_ADDRESS_SANITIZER})
    
##################################################################################################################################

    # [Add executable (target testing)]
    add_executable(${TEST_EXECUTABLE2} ${CMAKE_SOURCE_DIR}/test/MatTest.cpp)
    
    # [Set target include directories]
    target_include_directories(${TEST_EXECUTABLE2} PUBLIC ${PUBLIC_INCLUDES})
    
    # [Link target libraries]
    target_link_libraries(${TEST_EXECUTABLE2} PRIVATE GTest::gtest GTest::gtest_main CoolDiff)

    # [Set compile options]
    if(${ENABLE_COMPILER_WARNINGS})
        target_compile_options(${TEST_EXECUTABLE2} PRIVATE ${COMPILE_WARNING_FLAGS})
    endif()
    target_compile_options(${TEST_EXECUTABLE2} PRIVATE ${COMPILE_PERMISSION_FLAGS})
    target_compile_options(${TEST_EXECUTABLE2} PRIVATE ${HOOK_ADDRESS_SANITIZER})

    # [Set link options]
    target_link_options(${TEST_EXECUTABLE2} PRIVATE ${HOOK_ADDRESS_SANITIZER})
endif()

gtest_discover_tests(${TEST_EXECUTABLE})
gtest_discover_tests(${TEST_EXECUTABLE2})
