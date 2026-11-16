cmake_minimum_required(VERSION 3.13.4)

# 1. Set the target name
set(TARGET "program")

# 2. Source files
set(TARGET_SRC ${TARGET}.cc)

# 3. Find deal.II
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

# 7. -DHANGING_NODES_OVERLAY=ON: put overlay/ in front of the installed deal.II
# headers, so that Portable::FEEvaluation resolves hanging nodes with the
# batched kernel (batch size 1) instead of the cell-wise one, without
# rebuilding deal.II (all of it is instantiated in this program).
option(HANGING_NODES_OVERLAY "Use the batched hanging-node kernel through overlay/" OFF)
if(HANGING_NODES_OVERLAY)
  target_include_directories(${TARGET} BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/overlay)
  target_compile_definitions(${TARGET} PRIVATE HANGING_NODES_OVERLAY)
  message(STATUS "hanging-node kernel: batched (overlay/)")
else()
  message(STATUS "hanging-node kernel: cell-wise (installed deal.II)")
endif()
