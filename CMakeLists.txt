cmake_minimum_required(VERSION 3.26)
project(HinaPE
        LANGUAGES CXX
        VERSION 0.1.0
)

option(HINAPE_BUILD_TESTS "Build tests" ON)
option(HINAPE_BUILD_EXAMPLES "Build examples" ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(engine)

if (HINAPE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif ()

if (HINAPE_BUILD_EXAMPLES)
    include(cmake/setup_vulkan_visualizer.cmake)
    add_executable(HinaPEApp main.cpp)
    target_link_libraries(HinaPEApp PRIVATE HinaPE)
    use_vulkan_visualizer(HinaPEApp)
    add_subdirectory(examples)
endif ()