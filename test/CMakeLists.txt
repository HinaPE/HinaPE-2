cmake_minimum_required(VERSION 3.26)

if (NOT TARGET HinaPE)
    message(FATAL_ERROR "HinaPE target not found. Please add the main project before adding tests.")
endif ()

include(FetchContent)
FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.10.0
        GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)

add_executable(test_xpbd cloth/test_xpbd.cpp)
target_link_libraries(test_xpbd PRIVATE HinaPE Catch2::Catch2WithMain)

add_test(NAME xpbd COMMAND test_xpbd)