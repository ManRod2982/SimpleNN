cmake_minimum_required(VERSION 3.10)
project(simple_nn)

add_executable(simple_nn)

target_sources(simple_nn 
    PRIVATE
    src/main.cc
    src/simple_nn.cc
)

find_package(Catch2 REQUIRED)
add_executable(tests tests/test.cpp)
target_link_libraries(tests Catch2::Catch2)

include(CTest)
include(Catch)
catch_discover_tests(tests)