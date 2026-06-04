cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(NextHAM_Preprocessing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")

# Find LibTorch. Prefer caller-provided CMAKE_PREFIX_PATH, then common local layouts.
if(NOT CMAKE_PREFIX_PATH)
  foreach(TORCH_CANDIDATE
      "${CMAKE_CURRENT_SOURCE_DIR}/../../pkg/libtorch"
      "${CMAKE_CURRENT_SOURCE_DIR}/../../../libtorch")
    if(EXISTS "${TORCH_CANDIDATE}/share/cmake/Torch/TorchConfig.cmake")
      list(APPEND CMAKE_PREFIX_PATH "${TORCH_CANDIDATE}")
    endif()
  endforeach()
endif()
find_package(Torch REQUIRED)

# Find Eigen3. The project does not require linking, only headers.
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../pkg/eigen-3.4.0" CACHE PATH "Eigen3 include directory")
if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Dense")
  foreach(EIGEN_CANDIDATE
      "${CMAKE_CURRENT_SOURCE_DIR}/../../../eigen-3.4.0"
      "$ENV{CONDA_PREFIX}/include/eigen3"
      "/usr/include/eigen3"
      "/usr/local/include/eigen3")
    if(EXISTS "${EIGEN_CANDIDATE}/Eigen/Dense")
      set(EIGEN3_INCLUDE_DIR "${EIGEN_CANDIDATE}")
      break()
    endif()
  endforeach()
endif()
if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Dense")
  message(FATAL_ERROR "Eigen3 headers not found. Set -DEIGEN3_INCLUDE_DIR=/path/to/eigen3")
endif()
include_directories(${EIGEN3_INCLUDE_DIR})

# Find OpenMP
find_package(OpenMP REQUIRED)

# Add Source Files
set(SOURCES
    main.cpp
    stru_parser.cpp
    csr_parser.cpp
    block_parser.cpp
    neighbor_list.cpp
    transformer.cpp
)

add_executable(nextham_preprocess ${SOURCES})

# Target Properties
target_link_libraries(nextham_preprocess "${TORCH_LIBRARIES}" OpenMP::OpenMP_CXX)
set_property(TARGET nextham_preprocess PROPERTY CXX_STANDARD 17)

# Set runtime paths for shared libraries
if(UNIX)
  set_target_properties(nextham_preprocess PROPERTIES INSTALL_RPATH "${CMAKE_PREFIX_PATH}/lib")
endif()
