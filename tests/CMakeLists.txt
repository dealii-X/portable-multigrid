cmake_minimum_required(VERSION 3.13.4)

# Testsuite in deal.II's format: every name.cc in a subdirectory with a
# matching name[.variant].output is a test; deal.II's run_test.sh runs it
# (mpirun=N variants included) and compares the output with numdiff.
#
# Configured from the top-level CMakeLists.txt, or on its own:
#
#   cmake -S tests -B build-tests -DDEAL_II_DIR=<deal.II install>
#   ctest --test-dir build-tests -j<N>

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(TESTSUITE_STANDALONE TRUE)
  find_package(deal.II 9.8.0 REQUIRED
    HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
  )
endif()

# MPI launcher of the mpirun=N variants; deal.II's (mpiexec) by default.
# JUPITER: -DTESTSUITE_MPIEXEC=srun
#          -DTESTSUITE_MPIEXEC_PREFLAGS="--mpi=pspmix;--cpu-bind=cores;--gpu-bind=closest"
set(TESTSUITE_MPIEXEC "" CACHE STRING "MPI launcher for the tests")
set(TESTSUITE_MPIEXEC_PREFLAGS "" CACHE STRING "Flags after the rank count")
if(NOT "${TESTSUITE_MPIEXEC}" STREQUAL "")
  set(DEAL_II_MPIEXEC "${TESTSUITE_MPIEXEC}")
  set(DEAL_II_MPIEXEC_PREFLAGS ${TESTSUITE_MPIEXEC_PREFLAGS})
endif()

if(TESTSUITE_STANDALONE)
  deal_ii_initialize_cached_variables()
  project(portable_multigrid_testsuite CXX)
  enable_testing()
endif()

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

# Portable::internal::batched in deal.II: tests with
# "with_portable_batched=on" in the name of their output file only run then
# (Copy::Portable::MGTwoLevelTransfer)
if(NOT DEFINED DEAL_II_WITH_PORTABLE_BATCHED)
  set(DEAL_II_WITH_PORTABLE_BATCHED OFF)
  set(_header "${DEAL_II_PATH}/${DEAL_II_INCLUDE_RELDIR}/deal.II/matrix_free/portable_tensor_product_kernels.h")
  if(EXISTS "${_header}")
    file(STRINGS "${_header}" _batched REGEX "namespace batched")
    if(_batched)
      set(DEAL_II_WITH_PORTABLE_BATCHED ON)
    endif()
  endif()
endif()
message(STATUS "Portable::internal::batched: ${DEAL_II_WITH_PORTABLE_BATCHED}")

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

foreach(category kernels multigrid operators)
  add_subdirectory(${category})
endforeach()
