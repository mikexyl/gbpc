option(GBPC_BUILD_TESTS "Build tests" ON)

if(GBPC_BUILD_TESTS)
  add_executable(test_gbp_marg test_gbp_marg.cpp)
  target_include_directories(test_gbp_marg PRIVATE ${INCLUDES})
  target_link_libraries(test_gbp_marg ${PROJECT_NAME} Boost::program_options)

  # add_executable(test_connectivity
  # test_connectivity.cpp)
  # target_link_libraries(test_connectivity
  # PUBLIC
  # gbpc
  # GSL::gsl
  # matplot)
endif()
