cmake_minimum_required(VERSION 3.13.4)

# Testsuite (tests/):
#
#   cmake -S . -B build-tests -DDEAL_II_DIR=<deal.II install>
#   ctest --test-dir build-tests -j<N>
#
# With -DPORTABLE_MULTIGRID_PERFORMANCE_TESTS=ON also the performance tests
# (label "performance", see performance_tests/CMakeLists.txt). The drivers in
# source/ and the other benchmarks in performance_tests/ are separate CMake
# projects.

find_package(deal.II 9.8.0 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
)

deal_ii_initialize_cached_variables()
project(portable_multigrid CXX)
enable_testing()

option(PORTABLE_MULTIGRID_PERFORMANCE_TESTS "Performance tests with a stored baseline" OFF)

add_subdirectory(tests)
if(PORTABLE_MULTIGRID_PERFORMANCE_TESTS)
  add_subdirectory(performance_tests)
endif()
