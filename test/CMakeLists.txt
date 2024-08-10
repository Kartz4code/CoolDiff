# Add executable (target testing)
add_executable(OpsTest ${CMAKE_SOURCE_DIR}/test/OpsTest.cpp)
# Set target include directories
target_include_directories(OpsTest PUBLIC ${INCLUDES})	
# Set compile options
target_compile_options(OpsTest PRIVATE -fpermissive)
if(${BUILD_TYPE} STREQUAL Debug)
	target_compile_options(OpsTest PRIVATE -fsanitize=address)
	target_link_options(OpsTest PRIVATE -fsanitize=address)
endif()
target_link_libraries(OpsTest PRIVATE gtest_main CoolDiff)
