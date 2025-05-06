option(GBPC_BUILD_TESTS "Build tests" ON)

if(GBPC_BUILD_TESTS)
  find_package(SFML REQUIRED)

  # fetch tgui
  find_package(TGUI QUIET)
  if(NOT TGUI_FOUND)
    message(STATUS "TGUI not found, fetching...")
    include(FetchContent)
    FetchContent_Declare(
      tgui
      GIT_REPOSITORY https://github.com/texus/TGUI.git
      GIT_TAG v1.9.0
    )
    FetchContent_MakeAvailable(tgui)
  endif()

  # add_executable(test_gtsam
  #   test_gtsam.cpp)
  # target_include_directories(test_gtsam PRIVATE ${INCLUDES} ${GTSAM_INCLUDE_DIRS})
  # target_link_libraries(test_gtsam PRIVATE gbpc
  #   sfml-graphics sfml-window sfml-system
  #   tgui)

  add_executable(test_sci test_sci.cpp)
  target_include_directories(test_sci PRIVATE ${INCLUDES} ${GTSAM_INCLUDE_DIRS})
  target_link_libraries(test_sci PRIVATE gbpc
    sfml-graphics sfml-window sfml-system
    tgui)

  # add_executable(test_connectivity
  # test_connectivity.cpp)
  # target_link_libraries(test_connectivity
  # PUBLIC
  # gbpc
  # GSL::gsl
  # matplot)
endif()
