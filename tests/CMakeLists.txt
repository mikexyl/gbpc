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

    find_package(GTSAM QUIET)
    if(NOT GTSAM_FOUND)
        message(STATUS "GTSAM not found. Skipping test_gtsam.")
    else ()
        add_executable(test_gtsam
            test_gtsam.cpp)
        target_include_directories(test_gtsam PRIVATE ${INCLUDES} ${GTSAM_INCLUDE_DIRS})
        target_link_libraries(test_gtsam PRIVATE gbpc gtsam)
    endif ()

    find_package(GSL REQUIRED)
    include(FetchContent)
    FetchContent_Declare(
        matplotplusplus
        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus.git
        GIT_TAG v1.2.1
    )
    FetchContent_MakeAvailable(matplotplusplus)

    add_executable(test_connectivity
        test_connectivity.cpp)
    target_link_libraries(test_connectivity
        PUBLIC
        gbpc
        GSL::gsl
        matplot)
endif()