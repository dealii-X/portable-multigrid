cmake_minimum_required(VERSION 3.13.4)

# Testsuite in deal.II's format: every name.cc in a subdirectory with a
# matching name[.variant].output is a test; deal.II's run_test.sh runs it
# (mpirun=N variants included) and compares the output with numdiff.
#
#   cmake -S testsuite -B build-testsuite -DDEAL_II_DIR=<deal.II install>
#   cd build-testsuite && ctest -j<N>
#
# To be moved to tests/ once the benchmark drivers there have moved to
# performance_tests/ (#43).

find_package(deal.II 9.8.0 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
)

deal_ii_initialize_cached_variables()
project(portable_multigrid_testsuite CXX)
enable_testing()

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

foreach(category operators)
  add_subdirectory(${category})
endforeach()
