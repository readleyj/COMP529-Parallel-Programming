cmake_minimum_required(VERSION 3.20)
project(code)

find_package(MPI)

set(CMAKE_CXX_STANDARD 14)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(code
        bfs_main.cpp)

#target_link_libraries(code
#        PRIVATE
#        MPI_C)

target_link_libraries(code PRIVATE ${MPI_CXX_LIBRARIES})