cmake_minimum_required(VERSION 3.26)
project(pd_cloth_backend LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

get_filename_component(workspace "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(source "${workspace}/dependencies/libuipc")
set(native "${workspace}/dependencies/pyuipc-v0.0.28/uipc/_native")
set(PD_UIPC_BUILD_DEPS "" CACHE PATH "Headers and source trees used by the local libuipc build")
foreach(lib fmt spdlog cpptrace102 eigen json cppitertools magic_enum)
  if(NOT IS_DIRECTORY "${PD_UIPC_BUILD_DEPS}/${lib}")
    message(FATAL_ERROR "Missing ${lib} under PD_UIPC_BUILD_DEPS")
  endif()
endforeach()

set(FMT_TEST OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
set(CPPTRACE_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(CPPTRACE_GET_SYMBOLS_WITH_LIBDWARF OFF CACHE BOOL "" FORCE)
set(CPPTRACE_GET_SYMBOLS_WITH_LIBBACKTRACE OFF CACHE BOOL "" FORCE)
set(CPPTRACE_GET_SYMBOLS_WITH_ADDR2LINE ON CACHE BOOL "" FORCE)
add_subdirectory("${PD_UIPC_BUILD_DEPS}/fmt" deps_fmt EXCLUDE_FROM_ALL)
add_subdirectory("${PD_UIPC_BUILD_DEPS}/spdlog" deps_spdlog EXCLUDE_FROM_ALL)
add_subdirectory("${PD_UIPC_BUILD_DEPS}/cpptrace102" deps_cpptrace EXCLUDE_FROM_ALL)
find_package(CUDAToolkit REQUIRED)

foreach(lib core geometry io)
  add_library(uipc_${lib} SHARED IMPORTED)
  set_target_properties(uipc_${lib} PROPERTIES IMPORTED_LOCATION "${native}/libuipc_${lib}.so")
endforeach()

file(GLOB_RECURSE cuda_sources CONFIGURE_DEPENDS
  "${source}/src/backends/cuda/*.cu" "${source}/src/backends/cuda/*.cpp")
file(GLOB common_sources CONFIGURE_DEPENDS "${source}/src/backends/common/*.cpp")
add_library(uipc_backend_cuda MODULE ${cuda_sources} ${common_sources})
target_compile_definitions(uipc_backend_cuda PRIVATE UIPC_BACKEND_EXPORT_DLL=1
  UIPC_RUNTIME_CHECK=1 UIPC_VERSION_MAJOR=0 UIPC_VERSION_MINOR=0 UIPC_VERSION_PATCH=28
  "UIPC_PROJECT_DIR=R\"(${source})\"" "UIPC_BACKEND_DIR=R\"(${source}/src/backends)\""
  "UIPC_BACKEND_NAME=R\"(cuda)\"")
target_include_directories(uipc_backend_cuda PRIVATE
  "${source}/include" "${source}/src" "${source}/src/backends/cuda"
  "${source}/src/backends/cuda/cuda_tool" "${PD_UIPC_BUILD_DEPS}/eigen"
  "${PD_UIPC_BUILD_DEPS}/json/include" "${PD_UIPC_BUILD_DEPS}"
  "${PD_UIPC_BUILD_DEPS}/magic_enum/include")
target_link_libraries(uipc_backend_cuda PRIVATE uipc_core uipc_geometry uipc_io
  fmt::fmt spdlog::spdlog cpptrace::cpptrace CUDA::cudart ${CMAKE_DL_LIBS})
target_compile_options(uipc_backend_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;--extended-lambda;--diag-suppress=554>)
set_target_properties(uipc_backend_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/native")
# libuipc resolves both the CUDA backend and its CPU scene checker via module_dir.
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/native")
file(CREATE_LINK "${native}/libuipc_sanity_check.so"
  "${CMAKE_BINARY_DIR}/native/libuipc_sanity_check.so" SYMBOLIC)
foreach(file IN LISTS cuda_sources common_sources)
  file(RELATIVE_PATH relative "${source}" "${file}")
  set_source_files_properties("${file}" PROPERTIES COMPILE_DEFINITIONS
    "UIPC_RELATIVE_SOURCE_FILE=R\"(${relative})\"")
endforeach()
