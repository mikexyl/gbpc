option(GBPC_BUILD_DEMO "Build demo" ON)

if (GBPC_BUILD_DEMO)
    find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)
    if(NOT SFML_FOUND)
        FetchContent_Declare(
                SFML
                GIT_REPOSITORY https://github.com/SFML/SFML.git
                GIT_TAG 2.6.1  # Use the desired version or branch
        )
        FetchContent_MakeAvailable(SFML)
    endif()

    # fetch tgui
        FetchContent_Declare(
                tgui
                GIT_REPOSITORY https://github.com/texus/TGUI.git
                GIT_TAG v1.4.1
        )
        set(TGUI_BACKEND SFML_GRAPHICS)
        FetchContent_MakeAvailable(tgui)


    add_executable(demo
            demo.cpp)
    target_include_directories(demo
            PRIVATE
            ${INCLUDES})
    target_link_libraries(demo
            PRIVATE
            sfml-graphics sfml-window sfml-system
            tgui
            gbpc)

endif ()