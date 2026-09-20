cmake_minimum_required(VERSION 3.13.4)

# 1. Set the target name
set(TARGET "program")

# 2. Source files
set(TARGET_SRC ${TARGET}.cc)

# 3. Find deal.II
#
# The tensor-core columns in this benchmark's table only fill in when this
# ends up pointed at a deal.II built against a CUDA-enabled Kokkos -- with
# a Serial/OpenMP-only Kokkos, program.cc still builds and runs, it just
# reports the Kokkos (vmult_bk4) columns only. See tests/
# vector_laplace_tensor_core/CMakeLists.txt for the same note.
find_package(deal.II 9.8.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "Could not locate deal.II.")
endif()

# 4. Initialize and Project setup
deal_ii_initialize_cached_variables()
project(${TARGET} CXX)

# 5. Autopilot handles linking, flags (-O3), and Kokkos/MPI setup
deal_ii_invoke_autopilot()

# 6. Include Directories
target_include_directories(${TARGET} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
