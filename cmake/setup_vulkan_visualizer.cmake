# Vulkan Visualizer integration helper
# Repository: https://github.com/HinaPE/vulkan-visualizer
# Usage (example):
#   option(HINAPE_WITH_VULKAN_VISUALIZER "Enable Vulkan visualizer integration" ON)
#   if (HINAPE_WITH_VULKAN_VISUALIZER)
#       include(cmake/steup_vulkan_visualizer.cmake)
#       use_vulkan_visualizer(<your_target>)
#   endif ()
# After inclusion, if the visualizer library is found / fetched, the macro
# HINAPE_HAVE_VULKAN_VISUALIZER will be defined for the target passed to use_vulkan_visualizer.

include_guard(GLOBAL)

# Version / tag of the external project (adjust if needed)
set(HINAPE_VULKAN_VISUALIZER_TAG "master" CACHE STRING "vulkan-visualizer git tag or branch to fetch when fallback is used")
option(HINAPE_VULKAN_VISUALIZER_FETCH "Allow FetchContent fallback for vulkan-visualizer" ON)

# Allow user to point to a pre-installed package root manually (expected to contain a CMake package config or add_subdirectory content)
set(HINAPE_VULKAN_VISUALIZER_ROOT "" CACHE PATH "Path to existing vulkan-visualizer checkout/install (optional)")

# Try to locate a CONFIG package first if user has installed one.
if (NOT TARGET VulkanVisualizer::vulkan_visualizer AND NOT HINAPE_VULKAN_VISUALIZER_ROOT)
    find_package(vulkan_visualizer CONFIG QUIET)
endif ()

# If user supplied a root path, try add_subdirectory on it (assuming it has a top-level CMakeLists)
if (NOT TARGET VulkanVisualizer::vulkan_visualizer AND HINAPE_VULKAN_VISUALIZER_ROOT)
    if (EXISTS "${HINAPE_VULKAN_VISUALIZER_ROOT}/CMakeLists.txt")
        add_subdirectory("${HINAPE_VULKAN_VISUALIZER_ROOT}" _vulkan_visualizer_external)
    else ()
        message(FATAL_ERROR "HINAPE_VULKAN_VISUALIZER_ROOT specified but no CMakeLists.txt found: ${HINAPE_VULKAN_VISUALIZER_ROOT}")
    endif ()
endif ()

# FetchContent fallback
if (NOT TARGET VulkanVisualizer::vulkan_visualizer AND HINAPE_VULKAN_VISUALIZER_FETCH)
    include(FetchContent)
    FetchContent_Declare(hina_vulkan_visualizer
        GIT_REPOSITORY https://github.com/HinaPE/vulkan-visualizer.git
        GIT_TAG        ${HINAPE_VULKAN_VISUALIZER_TAG}
        GIT_SHALLOW    TRUE
        FIND_PACKAGE_ARGS
    )
    FetchContent_MakeAvailable(hina_vulkan_visualizer)
endif ()

# Attempt to create an imported / alias target if upstream does not define a namespaced one.
# We check for plausible targets. Adjust as repository evolves.
if (NOT TARGET VulkanVisualizer::vulkan_visualizer)
    if (TARGET vulkan_visualizer)
        add_library(VulkanVisualizer::vulkan_visualizer ALIAS vulkan_visualizer)
    elseif (TARGET vkviz) # hypothetical simple target name
        add_library(VulkanVisualizer::vulkan_visualizer ALIAS vkviz)
    endif ()
endif ()

if (NOT TARGET VulkanVisualizer::vulkan_visualizer)
    message(STATUS "vulkan-visualizer target not found (yet). You can: \n"
                   "  * Provide HINAPE_VULKAN_VISUALIZER_ROOT=path/to/clone \n"
                   "  * Or keep HINAPE_VULKAN_VISUALIZER_FETCH=ON for auto-fetch (if the repo has a proper CMakeLists).\n"
                   "Proceeding without visualizer.")
endif ()

function(use_vulkan_visualizer target)
    if (NOT TARGET ${target})
        message(FATAL_ERROR "use_vulkan_visualizer called with unknown target `${target}`")
    endif ()
    if (TARGET VulkanVisualizer::vulkan_visualizer)
        target_link_libraries(${target} PUBLIC VulkanVisualizer::vulkan_visualizer)
        target_compile_definitions(${target} PUBLIC HINAPE_HAVE_VULKAN_VISUALIZER=1)
    else ()
        message(WARNING "use_vulkan_visualizer: VulkanVisualizer::vulkan_visualizer not available; feature disabled for `${target}`")
    endif ()
endfunction()

