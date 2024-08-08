cmake_minimum_required(VERSION 3.16)
project(gbpc VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_definitions(-march=native)
add_compile_options(-Wno-sign-compare)

find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(TBB REQUIRED)
find_package(fmt REQUIRED)

find_package(GSL REQUIRED)
include(FetchContent)

find_package(Matplot++ QUIET)
if (NOT Matplot++_FOUND)
    FetchContent_Declare(
            matplotplusplus
            GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus.git
            GIT_TAG master)
    FetchContent_MakeAvailable(matplotplusplus)
endif()

set(INCLUDES
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${GTSAM_INCLUDE_DIR}
)

set(DEPENDENCIES
        PUBLIC
        Eigen3::Eigen
        gtsam
        TBB::tbb
        GSL::gsl
        fmt::fmt
)

add_library(gbpc SHARED
        src/gaussian.cpp
        src/gbpc.cpp)
target_include_directories(gbpc
        ${INCLUDES})
target_link_libraries(gbpc
        ${DEPENDENCIES})

# build tests
add_subdirectory(tests)

install(TARGETS gbpc
        EXPORT gbpcTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

install(DIRECTORY include/gbpc
        DESTINATION include)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/gbpc/gbpcConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY AnyNewerVersion
)

export(TARGETS gbpc
        NAMESPACE gbpc::
        FILE "${CMAKE_CURRENT_BINARY_DIR}/gbpc/gbpcTargets.cmake")

set(ConfigPackageLocation lib/cmake/gbpc_net)
install(EXPORT gbpcTargets
        FILE gbpcTargets.cmake
        NAMESPACE gbpc::
        DESTINATION ${ConfigPackageLocation})

# Use configure_file to process your template file
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gbpcConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/gbpcConfig.cmake"
        @ONLY
)

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/gbpcConfig.cmake"
        DESTINATION lib/cmake/gbpc
)
