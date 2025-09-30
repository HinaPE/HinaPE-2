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
