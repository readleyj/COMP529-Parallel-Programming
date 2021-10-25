include_directories("." ${PROJECT_SOURCE_DIR}/lib)
add_executable(mcubes main.cpp)
target_link_libraries (mcubes MeshReconstruction)