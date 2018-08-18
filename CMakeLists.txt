cmake_minimum_required(VERSION 2.6)
project(gauss_newton)

add_compile_options(-std=c++11)

find_package(OpenCV 3.0 QUIET)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(gauss_newton main.cpp)
target_link_libraries(gauss_newton ${OpenCV_LIBS})

install(TARGETS gauss_newton RUNTIME DESTINATION bin)
