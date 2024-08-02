option(GBPC_BUILD_TESTS "Build tests" ON)

if(GBPC_BUILD_TESTS)
    find_package(SFML REQUIRED)

    # fetch tgui
    find_package(TGUI REQUIRED)

  add_executable(test_gtsam
       test_gtsam.cpp)
  target_include_directories(test_gtsam PRIVATE ${INCLUDES} ${GTSAM_INCLUDE_DIRS})
  target_link_libraries(test_gtsam PRIVATE gbpc
        sfml-graphics sfml-window sfml-system
        tgui)

    find_package(Qt6 REQUIRED COMPONENTS Widgets Core Gui)
    # Enable automatic handling of moc files
set(CMAKE_AUTOMOC ON)

     add_executable(test_qt
       qt_visualizer.cpp)
   target_link_libraries(test_qt
       PUBLIC
       gbpc
       GSL::gsl)
    target_link_libraries(test_qt PRIVATE Qt6::Widgets Qt6::Core Qt6::Gui)
endif()
