# Performance tests (ctest label "performance"), from the top-level
# CMakeLists.txt with -DPORTABLE_MULTIGRID_PERFORMANCE_TESTS=ON:
#
#   ctest -L performance      run the drivers, compare with the baseline
#   ctest -LE performance     the testsuite only
#
# Each case runs a driver of poisson_cube_multigrid with --json and compares
# the result with baselines/<PERFORMANCE_MACHINE>/<case>.json
# (compare_baseline.py: same CG iterations, timings within the tolerance).
# A missing baseline only records; to store the result as the new baseline:
#
#   python3 compare_baseline.py <build>/performance_tests/<case>.json \
#     baselines/<machine>/<case>.json --update
#
# The other directories here are separate benchmark projects.

cmake_host_system_information(RESULT _host QUERY HOSTNAME)
set(PERFORMANCE_MACHINE "${_host}" CACHE STRING "Baseline set in performance_tests/baselines/")
set(PERFORMANCE_TOLERANCE "0.15" CACHE STRING "Allowed relative slowdown")
set(PERFORMANCE_N_RANKS "1" CACHE STRING "MPI ranks of the performance tests")
set(PERFORMANCE_MIN_SIZE "100000" CACHE STRING "Smallest problem (DoFs)")
set(PERFORMANCE_MAX_SIZE "2000000" CACHE STRING "Largest problem (DoFs)")
set(PERFORMANCE_DEGREES "2;4" CACHE STRING "Polynomial degrees")
set(PERFORMANCE_VARIANTS "poisson_cube_bk3" CACHE STRING "Driver variants")

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# the testsuite's launcher override (tests/CMakeLists.txt)
set(_mpiexec "${DEAL_II_MPIEXEC}")
set(_preflags ${DEAL_II_MPIEXEC_PREFLAGS})
if(DEFINED TESTSUITE_MPIEXEC AND NOT "${TESTSUITE_MPIEXEC}" STREQUAL "")
  set(_mpiexec "${TESTSUITE_MPIEXEC}")
  set(_preflags ${TESTSUITE_MPIEXEC_PREFLAGS})
endif()

set(MG_VARIANTS "${PERFORMANCE_VARIANTS}" CACHE STRING "Driver variants to build")
set(MG_SETUP_TARGET_TYPE RELEASE)
add_subdirectory(poisson_cube_multigrid)

set(_baselines "${CMAKE_CURRENT_SOURCE_DIR}/baselines/${PERFORMANCE_MACHINE}")
foreach(_variant ${PERFORMANCE_VARIANTS})
  foreach(_degree ${PERFORMANCE_DEGREES})
    set(_case "${_variant}_p${_degree}_np${PERFORMANCE_N_RANKS}")
    set(_json "${CMAKE_CURRENT_BINARY_DIR}/${_case}.json")

    # 5 pre- and post-smoothing steps, doubling mesh, no overlap
    add_test(NAME performance/${_case}.run
      COMMAND ${_mpiexec} ${DEAL_II_MPIEXEC_NUMPROC_FLAG} ${PERFORMANCE_N_RANKS} ${_preflags}
              $<TARGET_FILE:${_variant}> ${_degree} ${PERFORMANCE_MIN_SIZE}
              ${PERFORMANCE_MAX_SIZE} 5 5 d 0 --json ${_json})
    add_test(NAME performance/${_case}.compare
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/compare_baseline.py
              ${_json} ${_baselines}/${_case}.json --tolerance ${PERFORMANCE_TOLERANCE})

    set_tests_properties(performance/${_case}.run PROPERTIES
      LABELS performance RUN_SERIAL TRUE FIXTURES_SETUP ${_case})
    set_tests_properties(performance/${_case}.compare PROPERTIES
      LABELS performance FIXTURES_REQUIRED ${_case})
  endforeach()
endforeach()
