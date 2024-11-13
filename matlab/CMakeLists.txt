# Option to build MATLAB bindings
option(GBPC_BUILD_MATLAB "Build MATLAB bindings" OFF)

if(GBPC_BUILD_MATLAB)
    # Locate MATLAB and ensure it is available
    find_package(Matlab REQUIRED)

    # Define the MEX target as a shared library
    add_library(gaussianDiff SHARED gaussianDiff.cpp)

    # Include directories for MATLAB and your project
    target_include_directories(gaussianDiff
        PRIVATE
        ${INCLUDES}
        ${Matlab_INCLUDE_DIRS}
    )

    # Link MATLAB libraries and any project dependencies
    target_link_libraries(gaussianDiff
        PRIVATE
        gbpc
        ${Matlab_LIBRARIES}
    )

    # Set the output properties to match MATLAB's MEX file requirements
    set_target_properties(gaussianDiff PROPERTIES
        OUTPUT_NAME "gaussianDiff" # MATLAB expects the output name to match the function name
        SUFFIX ".${Matlab_MEX_EXTENSION}" # Set the MEX extension (platform-specific)
        PREFIX "" # Remove 'lib' prefix for MATLAB compatibility
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Matlab" # Optional: place output in a specific folder
    )

    # Optional: Install the MEX file if needed
    install(TARGETS gaussianDiff
        RUNTIME DESTINATION Matlab # For Windows
        LIBRARY DESTINATION Matlab # For macOS and Linux
        ARCHIVE DESTINATION Matlab # For static libraries (if applicable)
    )
endif()
