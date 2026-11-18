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

# MPI launcher of the mpirun=N variants; deal.II's (mpiexec) by default.
# JUPITER: -DTESTSUITE_MPIEXEC=srun
#          -DTESTSUITE_MPIEXEC_PREFLAGS="--mpi=pspmix;--cpu-bind=cores;--gpu-bind=closest"
set(TESTSUITE_MPIEXEC "" CACHE STRING "MPI launcher for the tests")
set(TESTSUITE_MPIEXEC_PREFLAGS "" CACHE STRING "Flags after the rank count")
if(NOT "${TESTSUITE_MPIEXEC}" STREQUAL "")
  set(DEAL_II_MPIEXEC "${TESTSUITE_MPIEXEC}")
  set(DEAL_II_MPIEXEC_PREFLAGS ${TESTSUITE_MPIEXEC_PREFLAGS})
endif()

deal_ii_initialize_cached_variables()
project(portable_multigrid_testsuite CXX)
enable_testing()

# Kokkos CUDA backend: tests with "with_kokkos_cuda=on" in the name of
# their output file only run then (nvcc kernels, tensor cores)
if(NOT DEFINED DEAL_II_WITH_KOKKOS_CUDA)
  if(NOT DEAL_II_TARGET_CONFIG_INCLUDED)
    include(${DEAL_II_TARGET_CONFIG})
    set(DEAL_II_TARGET_CONFIG_INCLUDED TRUE)
  endif()
  set(DEAL_II_WITH_KOKKOS_CUDA OFF)
  set(_kokkos_dirs)
  foreach(_target dealii::interface_kokkos Kokkos::kokkoscore Kokkos::kokkos)
    if(TARGET ${_target})
      get_target_property(_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
      if(_dirs)
        list(APPEND _kokkos_dirs ${_dirs})
      endif()
    endif()
  endforeach()
  foreach(_dir ${_kokkos_dirs})
    if(EXISTS "${_dir}/KokkosCore_config.h")
      file(STRINGS "${_dir}/KokkosCore_config.h" _cuda REGEX "^#define KOKKOS_ENABLE_CUDA$")
      if(_cuda)
        set(DEAL_II_WITH_KOKKOS_CUDA ON)
      endif()
    endif()
  endforeach()
endif()
message(STATUS "Kokkos CUDA backend: ${DEAL_II_WITH_KOKKOS_CUDA}")

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

foreach(category kernels operators)
  add_subdirectory(${category})
endforeach()
