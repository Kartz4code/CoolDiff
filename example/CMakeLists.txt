#[[
/**
* @file example/CMakeLists.txt
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

if(${BUILD_EXAMPLE})
    #[[
        Gaussian example
    ]]

    # Add executable (target testing)
    add_executable(Gaussian ${CMAKE_SOURCE_DIR}/example/Gaussian.cpp)
    # Set target include directories
    target_include_directories(Gaussian PUBLIC ${INCLUDES})
    # Set compile options
    target_compile_options(Gaussian PRIVATE -fpermissive)
    if(${BUILD_TYPE} STREQUAL Debug)
        target_compile_options(Gaussian PRIVATE -fno-omit-frame-pointer
                                                -fsanitize=address)
        target_link_options(
            Gaussian
            PRIVATE
            -fno-omit-frame-pointer
            -fsanitize=address)
    endif()
    target_link_libraries(Gaussian PRIVATE CoolDiff)

    #[[
        MatrixArithmetic example
    ]]

    # Add executable (target testing)
    add_executable(MatrixArithmetic
                   ${CMAKE_SOURCE_DIR}/example/MatrixArithmetic.cpp)
    # Set target include directories
    target_include_directories(MatrixArithmetic PUBLIC ${INCLUDES})
    # Set compile options
    target_compile_options(MatrixArithmetic PRIVATE -fpermissive)
    if(${BUILD_TYPE} STREQUAL Debug)
        target_compile_options(MatrixArithmetic PRIVATE -fno-omit-frame-pointer
                                                        -fsanitize=address)
        target_link_options(
            MatrixArithmetic
            PRIVATE
            -fno-omit-frame-pointer
            -fsanitize=address)
    endif()
    target_link_libraries(MatrixArithmetic PRIVATE CoolDiff)
endif()
