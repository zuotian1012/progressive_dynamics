cmake_minimum_required(VERSION 3.16)
project(CIPCNativeBaseline LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CIPC_SOURCE_ROOT "" CACHE PATH "Unmodified upstream Codim-IPC source")
set(CIPC_DEPS_ROOT "" CACHE PATH "Downloaded upstream dependencies")
if(NOT EXISTS "${CIPC_SOURCE_ROOT}/Library/EXPORTER.cpp")
  message(FATAL_ERROR "CIPC_SOURCE_ROOT must contain the upstream Library")
endif()
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory("${CIPC_DEPS_ROOT}/kokkos" kokkos-build)
set(Kokkos_DEVICES SERIAL OPENMP)
set(Cabana_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(Cabana_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${CIPC_DEPS_ROOT}/cabana" cabana-build)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GLUT REQUIRED)
find_path(CHOLMOD_INCLUDE_DIR cholmod.h PATH_SUFFIXES suitesparse REQUIRED)
find_library(CHOLMOD_LIBRARY cholmod REQUIRED)
pybind11_add_module(JGSL NO_EXTRAS
  "${CIPC_SOURCE_ROOT}/Library/EXPORTER.cpp"
  "${CIPC_SOURCE_ROOT}/Library/Math/Distance/EVCTCD/CTCD.cpp")
target_compile_definitions(JGSL PRIVATE STORAGE_ENABLED_OPENMP ENABLE_FEM_SHELL CHOLMOD_DIRECT_SOLVER)
target_compile_options(JGSL PRIVATE -mfma -mbmi2 -mavx2 -Wno-unused-variable -Wno-sign-compare -Wno-reorder)
target_include_directories(JGSL PRIVATE
  "${CIPC_SOURCE_ROOT}/Library"
  "${CIPC_SOURCE_ROOT}/Externals/flat_hash_map"
  "${CIPC_SOURCE_ROOT}/Externals/meta/include"
  "${CIPC_DEPS_ROOT}/amgcl"
  ${CHOLMOD_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
target_link_libraries(JGSL PRIVATE Kokkos::kokkos cabanacore Eigen3::Eigen
  OpenMP::OpenMP_CXX ${CHOLMOD_LIBRARY} ${GLUT_LIBRARIES})
