cmake_minimum_required(VERSION 3.13.4)

# Standalone build of the deal.II tests
#   tests/matrix_free_kokkos/hanging_node_kernels_01
#   tests/matrix_free_kokkos/hanging_node_kernels_02
# of the deal.II branch portable-hanging-nodes-line-kernel (commit 3db80e0257:
# Portable::internal::resolve_hanging_nodes() with one thread per line)
# against an installed deal.II with the old kernels.
#
# matrix_free_kokkos/, tests.h and overlay/ are verbatim copies from that
# branch. The header in overlay/ is template-only, so putting it in front of
# the installed headers is all it takes; deal.II is not rebuilt.

find_package(deal.II 9.8.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "Could not locate deal.II.")
endif()

deal_ii_initialize_cached_variables()
project(check_hanging_node_kernels CXX)

foreach(test hanging_node_kernels_01 hanging_node_kernels_02)
  add_executable(${test} matrix_free_kokkos/${test}.cc)
  deal_ii_setup_target(${test})
  # -I overlay before deal.II's own include directories
  target_include_directories(${test} BEFORE PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/overlay
  )
endforeach()
