option(GBPC_BUILD_TESTS "Build tests" ON)

if(GBPC_BUILD_TESTS)
  find_package(Catch2 3 REQUIRED)
  add_executable(test_hellinger test_hellinger.cpp)
  target_link_libraries(test_hellinger PRIVATE gbpc Catch2::Catch2WithMain)
  add_test(NAME test_hellinger COMMAND test_hellinger)

  include(FetchContent)
  FetchContent_Declare(matplotplusplus
        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
        GIT_TAG origin/master) # or whatever tag you want
  FetchContent_GetProperties(matplotplusplus)
  if(NOT matplotplusplus_POPULATED)
      FetchContent_Populate(matplotplusplus)
      add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()

  add_executable(test_hellinger_plot test_hellinger_plot.cpp)
  target_link_libraries(test_hellinger_plot PRIVATE gbpc matplot)
endif()
