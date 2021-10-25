cmake_minimum_required (VERSION 3.0)
project (MeshReconstruction)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I /kuacc/apps/llvm-omp/include -fopenmp=libomp")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I /kuacc/apps/llvm-omp/include -fopenmp=libomp")
    # set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Parallel compilation.
if(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4250")
endif()

set(CMAKE_DEBUG_POSTFIX "d")

add_subdirectory(lib)
add_subdirectory(demo)

