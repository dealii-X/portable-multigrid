cmake_minimum_required(VERSION 3.13.4)

# All of it in one build:
#
#   cmake -S . -B build -DDEAL_II_DIR=<deal.II install> -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j<N>         the drivers, build/<group>/<dir>/program
#   cmake --build build --target performance_tests.poisson_cube_bk3
#   ctest --test-dir build -j<N>      the testsuite (tests/)
#
# -DPORTABLE_MULTIGRID_DRIVERS=OFF / -DPORTABLE_MULTIGRID_BENCHMARKS=OFF skip
# source/ / performance_tests/. With -DPORTABLE_MULTIGRID_PERFORMANCE_TESTS=ON
# also the performance tests (label "performance", see
# performance_tests/CMakeLists.txt). Each directory in source/ and
# performance_tests/ still builds on its own.

find_package(deal.II 9.8.0 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
)

deal_ii_initialize_cached_variables()
project(portable_multigrid CXX)
enable_testing()

option(PORTABLE_MULTIGRID_DRIVERS "Build the drivers in source/" ON)
option(PORTABLE_MULTIGRID_BENCHMARKS "Build the benchmarks in performance_tests/" ON)
option(PORTABLE_MULTIGRID_PERFORMANCE_TESTS "Performance tests with a stored baseline" OFF)

# not yet: source/raviart-thomas, performance_tests/raviart_thomas_operator
# (do not build against the current include/)
set(_drivers
  cph_multigrid
  domain_decomposition
  geometric_multigrid
  geometric_multigrid_bk3
  geometric_multigrid_neumann
  newton_elasticity
  ph_multigrid
  ph_multigrid_bk3
  plain_laplace_solver
  plain_laplace_solver_dg
  plain_solver
  plain_solver_bk3
  polynomial_multigrid
  polynomial_multigrid_bk3
)

# poisson_cube_multigrid: see PORTABLE_MULTIGRID_PERFORMANCE_TESTS
set(_benchmarks
  abstracted_batched_laplace
  bp3
  bp4
  bp_35
  comm_microbench
  elasticity_benchmark_kernel
  elasticity_operator
  elasticity_residual
  hanging_nodes_comparison
  hanging_nodes_operator
  mg_transfer_comparison
  performance
  poisson_cube
  poisson_cube_bk3
  poisson_cube_bk3_dg
  poisson_cube_bk3_fp64
  poisson_cube_bk3_fp64_dg
  poisson_cube_cpu
  poisson_cube_hybrid_gpu_cpu
  test-pop-coe
)

# as in their own CMakeLists.txt
set(_with_common
  performance
  poisson_cube
  poisson_cube_bk3
  poisson_cube_bk3_dg
  poisson_cube_bk3_fp64
  poisson_cube_bk3_fp64_dg
  poisson_cube_cpu
  poisson_cube_hybrid_gpu_cpu
  test-pop-coe
)
set(_source_performance timing_ph_mg)

# target <group>.<dir>, executable build/<group>/<dir>/program
function(portable_multigrid_add_program _group _dir)
  set(_name program)
  if(DEFINED _source_${_dir})
    set(_name ${_source_${_dir}})
  endif()
  set(_target ${_group}.${_dir})
  add_executable(${_target} ${CMAKE_SOURCE_DIR}/${_group}/${_dir}/${_name}.cc)
  deal_ii_setup_target(${_target})
  set_target_properties(${_target} PROPERTIES
    OUTPUT_NAME ${_name}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${_group}/${_dir})
  target_include_directories(${_target} PRIVATE
    ${CMAKE_SOURCE_DIR}/${_group}/${_dir}
    ${CMAKE_SOURCE_DIR}/include)
  if(_group STREQUAL "performance_tests" AND _dir IN_LIST _with_common)
    target_include_directories(${_target} PRIVATE
      ${CMAKE_SOURCE_DIR}/performance_tests/common)
  endif()
endfunction()

if(PORTABLE_MULTIGRID_DRIVERS)
  foreach(_dir ${_drivers})
    portable_multigrid_add_program(source ${_dir})
  endforeach()
endif()

if(PORTABLE_MULTIGRID_BENCHMARKS)
  foreach(_dir ${_benchmarks})
    portable_multigrid_add_program(performance_tests ${_dir})
  endforeach()
endif()

if(TARGET performance_tests.elasticity_residual)
  include(${CMAKE_SOURCE_DIR}/performance_tests/elasticity_residual/instantiations.cmake)
  elasticity_residual_instantiations(performance_tests.elasticity_residual)
endif()

add_subdirectory(tests)
if(PORTABLE_MULTIGRID_PERFORMANCE_TESTS)
  add_subdirectory(performance_tests)
endif()
