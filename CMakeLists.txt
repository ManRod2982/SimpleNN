cmake_minimum_required(VERSION 3.10)
project(simple_nn)

# Build the core library so other targets (exe, tests) can link to it
add_library(simple_nn_lib
    src/simple_nn.cc
    src/mnist_reader.cc
)

# The executable links the core library and provides the CLI
add_executable(simple_nn
    src/main.cc
)
target_link_libraries(simple_nn PRIVATE simple_nn_lib)
target_compile_definitions(simple_nn_lib PRIVATE $<$<CONFIG:Debug>:DEBUG>)


find_package(Catch2 REQUIRED)
add_executable(tests tests/test.cpp)
# Link tests with the library under test and Catch2
target_link_libraries(tests PRIVATE Catch2::Catch2 simple_nn_lib)

find_package(Eigen3 REQUIRED NO_MODULE) # Use Eigen3::Eigen target
target_link_libraries(simple_nn PRIVATE Eigen3::Eigen)
target_compile_definitions(simple_nn PRIVATE $<$<CONFIG:Debug>:DEBUG>)
target_link_libraries(tests PRIVATE Eigen3::Eigen)

include(CTest)
include(Catch)
catch_discover_tests(tests)