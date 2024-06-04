option(GBPC_BUILD_TESTS "Build tests" ON)

if(GBPC_BUILD_TESTS)
    add_executable(test_regression
        test_regression.cpp)
    target_include_directories(test_regression
        PRIVATE
        ${INCLUDES})
    target_link_libraries(test_regression
        PRIVATE
        gbpc)

    find_package(GTSAM REQUIRED)
    add_executable(test_gtsam
        test_gtsam.cpp)
    target_include_directories(test_gtsam PRIVATE ${GTSAM_INCLUDE_DIR} ${INCLUDES})
    target_link_libraries(test_gtsam PRIVATE gbpc gtsam)
endif()