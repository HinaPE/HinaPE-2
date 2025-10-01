cmake_minimum_required(VERSION 3.26)

if (NOT TARGET HinaPE)
    message(FATAL_ERROR "HinaPE target not found. Add the main project before tests.")
endif ()

include(FetchContent)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.10.0
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(Catch2)

include(../cmake/setup_tbb.cmake OPTIONAL)

function(hinape_add_cloth_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE HinaPE Catch2::Catch2WithMain)
    if (COMMAND use_tbb)
        use_tbb(${target})
    endif ()
endfunction()

hinape_add_cloth_test(test_xpbd cloth/test_xpbd.cpp)
add_test(NAME xpbd COMMAND test_xpbd)

hinape_add_cloth_test(bench_xpbd cloth/bench_xpbd.cpp)
add_test(NAME xpbd_bench COMMAND bench_xpbd --reporter console --benchmark-samples 30 --benchmark-warmup-time 0.2)

hinape_add_cloth_test(test_pd cloth/test_pd.cpp)
add_test(NAME pd COMMAND test_pd)

hinape_add_cloth_test(test_fem cloth/test_fem.cpp)
add_test(NAME fem COMMAND test_fem)

