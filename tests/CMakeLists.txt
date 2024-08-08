option(GBPC_BUILD_TESTS "Build tests" ON)

if(GBPC_BUILD_TESTS)
    find_package(SFML REQUIRED)

    # fetch tgui
    find_package(TGUI QUIET)
    if(NOT TGUI_FOUND)
        message(STATUS "TGUI not found, fetching from git")
        include(FetchContent)
        FetchContent_Declare(
            tgui
            GIT_REPOSITORY https://github.com/texus/TGUI.git
            GIT_TAG 1.x)
        set(TGUI_BACKEND SFML_GRAPHICS)
        FetchContent_MakeAvailable(tgui)
    endif()

  add_executable(test_gtsam
       test_gtsam.cpp)
  target_include_directories(test_gtsam PRIVATE ${INCLUDES} ${GTSAM_INCLUDE_DIRS})
  target_link_libraries(test_gtsam PRIVATE gbpc
        sfml-graphics sfml-window sfml-system
        tgui)

    find_package(Graphviz REQUIRED)
    find_package(Qt6 REQUIRED COMPONENTS Widgets Core Gui)
    # Enable automatic handling of moc files
    set(CMAKE_AUTOMOC ON)

    set(SOURCES graph_widget.cpp qt_visualizer.cpp)
    set(HEADERS graph_widget.h)

     add_executable(test_qt ${SOURCES} ${HEADERS})
   target_link_libraries(test_qt
       PUBLIC
       gbpc
       GSL::gsl)
    target_link_libraries(test_qt PRIVATE
        Qt6::Widgets
        Qt6::Core
        Qt6::Gui
        ${GRAPHVIZ_CDT_LIBRARY}
        ${GRAPHVIZ_GVC_LIBRARY}
        ${GRAPHVIZ_CGRAPH_LIBRARY}
        ${GRAPHVIZ_PATHPLAN_LIBRARY})
    target_include_directories(test_qt PRIVATE
        ${GRAPHVIZ_INCLUDE_DIRS})

    FetchContent_Declare(
        nanoflann
        GIT_REPOSITORY https://github.com/jlblancoc/nanoflann.git
        GIT_TAG v1.6.0)
    FetchContent_MakeAvailable(nanoflann)
    add_executable(swarm_game
            playground.h
            robot.h
           swarm_game.cpp)
     target_link_libraries(swarm_game PRIVATE
        gbpc
        nanoflann
        Qt6::Widgets
        Qt6::Core
        Qt6::Gui)
endif()
