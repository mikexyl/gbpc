option(GBPC_BUILD_TESTS "Build tests" ON)

if(GBPC_BUILD_TESTS)
  add_executable(test_gtsam
       test_gtsam.cpp)
  target_include_directories(test_gtsam PRIVATE ${INCLUDES} ${GTSAM_INCLUDE_DIRS})
  target_link_libraries(test_gtsam PRIVATE gbpc)

  #    add_executable(test_connectivity
  #      test_connectivity.cpp)
  #  target_link_libraries(test_connectivity
  #      PUBLIC
  #      gbpc
  #      GSL::gsl
  #      matplot)
endif()
