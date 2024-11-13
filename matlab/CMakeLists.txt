# Option to build MATLAB bindings
option(GBPC_BUILD_MATLAB "Build MATLAB bindings" OFF)

if(GBPC_BUILD_MATLAB)
    # Locate MATLAB and ensure it is available
    find_package(Matlab REQUIRED)

    # Define the MEX target as a shared library
    add_library(gbpc_matlab SHARED src/gbpc_mex.cpp)

    # Include directories for MATLAB and your project
    target_include_directories(gbpc_matlab
        PRIVATE
        ${INCLUDES}
        ${Matlab_INCLUDE_DIRS}
    )

    # Link MATLAB libraries and any project dependencies
    target_link_libraries(gbpc_matlab
        PRIVATE
        gbpc
        ${Matlab_LIBRARIES}
    )

    # Set the output properties to match MATLAB's MEX file requirements
    set_target_properties(gbpc_matlab PROPERTIES
        OUTPUT_NAME "gbpc_mex"                             # MATLAB expects the output name to match the function name
        SUFFIX ".${Matlab_MEX_EXTENSION}"                  # Set the MEX extension (platform-specific)
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/matlab" # Optional: place output in a specific folder
    )

    # Optional: Install the MEX file if needed
    install(TARGETS gbpc_matlab
        RUNTIME DESTINATION matlab                    # For Windows
        LIBRARY DESTINATION matlab                    # For macOS and Linux
        ARCHIVE DESTINATION matlab                    # For static libraries (if applicable)
    )
endif()
