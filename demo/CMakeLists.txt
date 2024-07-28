option(GBPC_BUILD_DEMO "Build demo" ON)

if (GBPC_BUILD_DEMO)
    FetchContent_Declare(
            SFML
            GIT_REPOSITORY https://github.com/SFML/SFML.git
            GIT_TAG 2.6.1  # Use the desired version or branch
    )
    FetchContent_MakeAvailable(SFML)

    add_executable(demo
            demo.cpp)
    target_include_directories(demo
            PRIVATE
            ${INCLUDES})
    target_link_libraries(demo
            PRIVATE
            sfml-graphics sfml-window sfml-system
            gbpc)

endif ()