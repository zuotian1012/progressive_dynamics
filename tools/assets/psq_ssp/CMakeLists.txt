cmake_minimum_required(VERSION 3.16)
project(psq_ssp LANGUAGES CXX)
set(PSQ_SSP_SOURCE "" CACHE PATH "Checkout of surface_multigrid_code with libigl submodule")
set(PSQ_EIGEN_INCLUDE "" CACHE PATH "Eigen include directory")
if(NOT EXISTS "${PSQ_SSP_SOURCE}/libigl/include/igl/read_triangle_mesh.h"
   OR NOT EXISTS "${PSQ_EIGEN_INCLUDE}/Eigen/Core")
  message(FATAL_ERROR "Set PSQ_SSP_SOURCE and PSQ_EIGEN_INCLUDE to the offline dependencies")
endif()
set(ssp_sources
  SSP_collapse_edge SSP_decimate SSP_midpoint SSP_qslim SSP_qslim_optimal_collapse_edge_callbacks
  SSP_vertexRemoval SSP_vertexRemoval_optimal_collapse_edge_callbacks always_try_never_care
  compute_barycentric compute_vertex_quadrics cotmatrix_dense get_collapse_onering_faces
  get_post_faces intersect_ordered joint_lscm mqwf_dense quasi_conformal_error
  query_fine_to_coarse remove_row remove_unreferenced_lessF remove_vector_element
  vector_area_matrix_size vector_mod)
list(TRANSFORM ssp_sources PREPEND "${PSQ_SSP_SOURCE}/src/")
list(TRANSFORM ssp_sources APPEND ".cpp")
find_package(Threads REQUIRED)
add_executable(psq_ssp main.cpp ${ssp_sources})
target_compile_features(psq_ssp PRIVATE cxx_std_17)
target_include_directories(psq_ssp PRIVATE "${PSQ_SSP_SOURCE}/src"
  "${PSQ_SSP_SOURCE}/libigl/include" "${PSQ_EIGEN_INCLUDE}")
target_link_libraries(psq_ssp PRIVATE Threads::Threads)
