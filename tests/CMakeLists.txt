option(GBPC_BUILD_TESTS "Build tests" ON)

if(GBPC_BUILD_TESTS)
    FetchContent_Declare(
                SFML
                GIT_REPOSITORY https://github.com/SFML/SFML.git
                GIT_TAG 2.6.1  # Use the desired version or branch
        )
        FetchContent_MakeAvailable(SFML)

    # fetch tgui
        FetchContent_Declare(
                tgui
                GIT_REPOSITORY https://github.com/texus/TGUI.git
                GIT_TAG v1.4.1
        )
        set(TGUI_BACKEND SFML_GRAPHICS)
        set(TGUI_BUILD_FRAMEWORK OFF)
        FetchContent_MakeAvailable(tgui)

  add_executable(test_gtsam
       test_gtsam.cpp)
  target_include_directories(test_gtsam PRIVATE ${INCLUDES} ${GTSAM_INCLUDE_DIRS})
  target_link_libraries(test_gtsam PRIVATE gbpc
        sfml-graphics sfml-window sfml-system
        tgui)

  #    add_executable(test_connectivity
  #      test_connectivity.cpp)
  #  target_link_libraries(test_connectivity
  #      PUBLIC
  #      gbpc
  #      GSL::gsl
  #      matplot)
endif()
