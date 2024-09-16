if (GBPC_BUILD_TESTS)
    find_package(SFML REQUIRED)

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

        add_executable(tgui_example
        main-SFML_GRAPHICS.cpp
        ManyDifferentWidgets.cpp)
        target_include_directories(tgui_example
        PRIVATE
        ${INCLUDES})

        target_link_libraries(tgui_example
        PRIVATE
        sfml-graphics sfml-window sfml-system
        tgui)

endif ()