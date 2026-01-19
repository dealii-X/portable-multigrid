cmake_minimum_required(VERSION 3.13.4)

# 1. Set the target name
set(TARGET "timing_ph_mg")

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

# 5. Define Macros for the Performance Driver
# These are required by your "performance_test_driver.h"
# add_definitions(-DENABLE_MPI)
# add_definitions(-DENABLE_PERFORMANCE_TESTS)
# add_definitions(-DTESTING_ENVIRONMENT=light) # Change to medium/heavy as needed

# 6. Autopilot handles linking, flags (-O3), and Kokkos/MPI setup
deal_ii_invoke_autopilot()

# 7. Include Directories
# This points to where your "multigrid/" and "operators/" folders are
target_include_directories(${TARGET} PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)