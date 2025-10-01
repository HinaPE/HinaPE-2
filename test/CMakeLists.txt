cmake_minimum_required(VERSION 3.26)

if (NOT TARGET HinaPE)
    message(FATAL_ERROR "HinaPE target not found. Add the main project before tests.")
endif ()

include(FetchContent)
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.10.0
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(Catch2)

add_executable(test_xpbd cloth/test_xpbd.cpp)
target_link_libraries(test_xpbd PRIVATE HinaPE Catch2::Catch2WithMain)
include(../cmake/setup_tbb.cmake OPTIONAL)
if (COMMAND use_tbb)
    use_tbb(test_xpbd)
endif ()
add_test(NAME xpbd COMMAND test_xpbd)

# Benchmarks: backend performance comparison for XPBD cloth
add_executable(bench_xpbd cloth/bench_xpbd.cpp)
target_link_libraries(bench_xpbd PRIVATE HinaPE Catch2::Catch2WithMain)
if (COMMAND use_tbb)
    use_tbb(bench_xpbd)
endif ()
add_test(NAME xpbd_bench COMMAND bench_xpbd --reporter console --benchmark-samples 30 --benchmark-warmup-time 0.2)

# PD tests
add_executable(test_pd cloth/test_pd.cpp)
target_link_libraries(test_pd PRIVATE HinaPE Catch2::Catch2WithMain)
if (COMMAND use_tbb)
    use_tbb(test_pd)
endif ()
add_test(NAME pd COMMAND test_pd)
