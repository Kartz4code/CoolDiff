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

if(${BUILD_EXAMPLES})
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
                                        # [Gaussian example]
    ########################################################################################################################
    ########################################################################################################################
    ######################################################################################################################## 

    # [Add executable (target testing)]
    add_executable(Gaussian ${CMAKE_SOURCE_DIR}/example/Gaussian.cpp)

    # [Set target include directories]
    target_include_directories(Gaussian PUBLIC ${INCLUDES})

    # [Link target libraries]
    target_link_libraries(Gaussian PRIVATE CoolDiff)

    # [Set compile options]
    if(${ENABLE_COMPILER_WARNINGS})
        target_compile_options(Gaussian PRIVATE ${WARNING_FLAGS})
    endif()
    target_compile_options(Gaussian PRIVATE ${PERMISSION_FLAGS})
    target_compile_options(Gaussian PRIVATE ${ATTACH_ADDRESS_SANITIZER})

    # [Set link options]
    target_link_options(Gaussian PRIVATE ${ATTACH_ADDRESS_SANITIZER})

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
                                        # [GaussNewton example]
    ########################################################################################################################
    ########################################################################################################################
    ######################################################################################################################## 

    # [Add executable (target testing)]
    set(SOURCES ${CMAKE_SOURCE_DIR}/example/GaussNewton/src/OracleMatrix.cpp
                ${CMAKE_SOURCE_DIR}/example/GaussNewton/src/OracleScalar.cpp
                ${CMAKE_SOURCE_DIR}/example/GaussNewton/src/GaussNewton.cpp
                ${CMAKE_SOURCE_DIR}/example/GaussNewton/src/Oracle.cpp)

    add_executable(GaussNewton ${CMAKE_SOURCE_DIR}/example/GaussNewton/main.cpp ${SOURCES})

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/GaussNewton/config/GaussNewtonData.hpp.in 
                    GaussNewtonData.hpp 
                    @ONLY)

    configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/GaussNewton/config/GaussNewtonData.py.in 
                    ${CMAKE_CURRENT_SOURCE_DIR}/GaussNewton/data/GaussNewtonData.py 
                    @ONLY)

    # [Set target include directories]
    target_include_directories(GaussNewton PUBLIC   ${INCLUDES} 
                                                    ${CMAKE_SOURCE_DIR}/example/GaussNewton/include
                                                    ${CMAKE_CURRENT_BINARY_DIR})

    # [Link target libraries]
    target_link_libraries(GaussNewton PRIVATE CoolDiff)

    # [Set compile options]
    if(${ENABLE_COMPILER_WARNINGS})
        target_compile_options(GaussNewton PRIVATE ${WARNING_FLAGS})
    endif()
    target_compile_options(GaussNewton PRIVATE ${PERMISSION_FLAGS})
    target_compile_options(GaussNewton PRIVATE ${ATTACH_ADDRESS_SANITIZER})

    # [Set link options]
    target_link_options(GaussNewton PRIVATE ${ATTACH_ADDRESS_SANITIZER})
endif()
