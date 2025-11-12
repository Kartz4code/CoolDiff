#[[
/**
* @file example/CMakeLists.txt
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
    target_include_directories(Gaussian PUBLIC ${PUBLIC_INCLUDES})

    # [Link target libraries]
    target_link_libraries(Gaussian PRIVATE CoolDiff)

    # [Set compile options]
    if(${ENABLE_COMPILER_WARNINGS})
        target_compile_options(Gaussian PRIVATE ${COMPILE_WARNING_FLAGS})
    endif()
    target_compile_options(Gaussian PRIVATE ${COMPILE_PERMISSION_FLAGS})
    target_compile_options(Gaussian PRIVATE ${HOOK_ADDRESS_SANITIZER})

    # [Set link options]
    target_link_options(Gaussian PRIVATE ${HOOK_ADDRESS_SANITIZER})

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
    target_include_directories(GaussNewton PUBLIC   ${PUBLIC_INCLUDES} 
                                                    ${CMAKE_SOURCE_DIR}/example/GaussNewton/include
                                                    ${CMAKE_CURRENT_BINARY_DIR})

    # [Link target libraries]
    target_link_libraries(GaussNewton PRIVATE CoolDiff)

    # [Set compile options]
    if(${ENABLE_COMPILER_WARNINGS})
        target_compile_options(GaussNewton PRIVATE ${COMPILE_WARNING_FLAGS})
    endif()
    target_compile_options(GaussNewton PRIVATE ${COMPILE_PERMISSION_FLAGS})
    target_compile_options(GaussNewton PRIVATE ${HOOK_ADDRESS_SANITIZER})

    # [Set link options]
    target_link_options(GaussNewton PRIVATE ${HOOK_ADDRESS_SANITIZER})

########################################################################################################################
########################################################################################################################
########################################################################################################################
                                    # [Neural network example]
########################################################################################################################
########################################################################################################################
######################################################################################################################## 

    # [Add executable (target testing)]
    set(SOURCES ${CMAKE_SOURCE_DIR}/example/NeuralNet/src/Layer.cpp)

    add_executable(NeuralNet ${CMAKE_SOURCE_DIR}/example/NeuralNet/main.cpp ${SOURCES})

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/NeuralNet/config/MNISTData.hpp.in MNISTData.hpp @ONLY)

    # Find and install csv-parser package
    find_package(csv QUIET)
    if(DEFINED csv_FOUND AND DEFINED csv_VERSION)
        message("[${CMAKE_BUILD_TYPE}][${PROJECT_NAME}] csv installed version ${csv_VERSION}")
    endif()
    if(NOT csv_FOUND)
        FetchContent_Declare(
        csv
        GIT_REPOSITORY https://github.com/vincentlaucsb/csv-parser.git
        GIT_TAG master
        GIT_SHALLOW TRUE
        ) 
        # Check if the dependency has already been downloaded and built
        FetchContent_GetProperties(csv)
        if(NOT csv_POPULATED)
            message("[${CMAKE_BUILD_TYPE}][${PROJECT_NAME}] Setting latest csv library")
            FetchContent_MakeAvailable(csv)
        endif()
    else()
        message("[${CMAKE_BUILD_TYPE}][${PROJECT_NAME}] csv library found in system")
    endif()

    # [Set target include directories]
    target_include_directories(NeuralNet PUBLIC ${PUBLIC_INCLUDES} 
                                                ${CMAKE_SOURCE_DIR}/example/NeuralNet/include
                                                ${CSV_INCLUDE_DIR}
                                                ${CMAKE_CURRENT_BINARY_DIR})

    # [Link target libraries]
    target_link_libraries(NeuralNet PUBLIC csv CoolDiff  gcov)

    # [Set compile options]
    if(${ENABLE_COMPILER_WARNINGS})
        target_compile_options(NeuralNet PRIVATE ${COMPILE_WARNING_FLAGS})
    endif()
    target_compile_options(NeuralNet PRIVATE ${COMPILE_PERMISSION_FLAGS})
    target_compile_options(NeuralNet PRIVATE ${HOOK_ADDRESS_SANITIZER})

    # [Set link options]
    target_link_options(NeuralNet PRIVATE ${HOOK_ADDRESS_SANITIZER})
endif()