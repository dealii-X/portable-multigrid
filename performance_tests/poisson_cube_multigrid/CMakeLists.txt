cmake_minimum_required(VERSION 3.13.4)

# One multigrid benchmark driver (program.cc), built as the variants that
# used to be separate copies:
#
#   poisson_cube_bk3          BK3, float V-cycle under double CG
#   poisson_cube_bk3_fp64     BK3, double V-cycle
#   poisson_cube_bk3_dg       FE_DGQ fine level (LaplaceOperatorDG), float V-cycle
#   poisson_cube_bk3_fp64_dg  FE_DGQ fine level, double V-cycle
#   poisson_cube_generic      Portable::LaplaceOperator, double, p-coarsening by 1
#
#   cmake -S . -B build -DDEAL_II_DIR=<deal.II install> [-DMG_VARIANTS="poisson_cube_bk3"]
#   ./build/poisson_cube_bk3 degree minsize maxsize n_pre n_post doubling overlap [--json file]

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(deal.II 9.8.0 REQUIRED
    HINTS ${deal.II_DIR} ${DEAL_II_DIR} $ENV{DEAL_II_DIR}
  )
  deal_ii_initialize_cached_variables()
  project(poisson_cube_multigrid CXX)
endif()

set(_all_variants
  poisson_cube_bk3
  poisson_cube_bk3_fp64
  poisson_cube_bk3_dg
  poisson_cube_bk3_fp64_dg
  poisson_cube_generic
)
set(MG_VARIANTS "${_all_variants}" CACHE STRING "Driver variants to build")

set(_defs_poisson_cube_bk3)
set(_defs_poisson_cube_bk3_fp64 MG_VCYCLE_DOUBLE)
set(_defs_poisson_cube_bk3_dg MG_DG)
set(_defs_poisson_cube_bk3_fp64_dg MG_VCYCLE_DOUBLE MG_DG)
set(_defs_poisson_cube_generic MG_VCYCLE_DOUBLE MG_GENERIC_OPERATOR MG_P_STEP=1)

foreach(_variant ${MG_VARIANTS})
  if(NOT _variant IN_LIST _all_variants)
    message(FATAL_ERROR "Unknown variant ${_variant}, choose from ${_all_variants}")
  endif()
  add_executable(${_variant} ${CMAKE_CURRENT_SOURCE_DIR}/program.cc)
  # MG_SETUP_TARGET_TYPE: RELEASE from performance_tests/, else CMAKE_BUILD_TYPE
  deal_ii_setup_target(${_variant} ${MG_SETUP_TARGET_TYPE})
  target_compile_definitions(${_variant} PRIVATE
    ${_defs_${_variant}} MG_VARIANT_NAME="${_variant}")
  target_include_directories(${_variant} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
  )
endforeach()
