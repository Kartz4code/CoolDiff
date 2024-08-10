if(${BUILD_TEST})
	# Add executable (target testing)
	add_executable(Gaussian ${CMAKE_SOURCE_DIR}/example/Gaussian.cpp)
	# Set target include directories
	target_include_directories(Gaussian PUBLIC ${INCLUDES})	
	# Set compile options
	target_compile_options(Gaussian PRIVATE -fpermissive)
	if(${BUILD_TYPE} STREQUAL Debug)
		target_compile_options(Gaussian PRIVATE -fsanitize=address)
		target_link_options(Gaussian PRIVATE -fsanitize=address)
	endif()
	target_link_libraries(Gaussian PRIVATE CoolDiff)
endif()