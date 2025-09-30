# Vulkan Visualizer integration helper
# Repository: https://github.com/HinaPE/vulkan-visualizer
# Usage (example):
#   option(HINAPE_WITH_VULKAN_VISUALIZER "Enable Vulkan visualizer integration" ON)
#   if (HINAPE_WITH_VULKAN_VISUALIZER)
#       include(cmake/setup_vulkan_visualizer.cmake)
#       use_vulkan_visualizer(<your_target>)
#   endif ()
# After inclusion, if the visualizer library is found / fetched, the macro
# HINAPE_HAVE_VULKAN_VISUALIZER will be defined for the target passed to use_vulkan_visualizer.

include_guard(GLOBAL)

# Version / tag of the external project (adjust if needed)
set(HINAPE_VULKAN_VISUALIZER_TAG "master" CACHE STRING "vulkan-visualizer git tag or branch to fetch when fallback is used")
option(HINAPE_VULKAN_VISUALIZER_FETCH "Allow FetchContent fallback for vulkan-visualizer" ON)

# Runtime DLL copy control (Windows)
option(HINAPE_VULKAN_VISUALIZER_COPY_DLLS "Copy vulkan-visualizer (and related) runtime DLLs next to targets on Windows" ON)
set(HINAPE_VULKAN_VISUALIZER_EXTRA_DLLS "" CACHE STRING "Semicolon-separated extra DLL absolute paths or relative paths to copy")

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

function(_vv_copy_runtime_dlls target)
    if (NOT WIN32)
        return()
    endif ()
    if (NOT HINAPE_VULKAN_VISUALIZER_COPY_DLLS)
        return()
    endif ()
    # Basic candidate targets to copy runtime from
    set(_vv_candidates)
    if (TARGET VulkanVisualizer::vulkan_visualizer)
        list(APPEND _vv_candidates VulkanVisualizer::vulkan_visualizer)
    endif ()
    if (TARGET SDL3::SDL3)
        list(APPEND _vv_candidates SDL3::SDL3)
    elseif (TARGET SDL3::SDL3-shared)
        list(APPEND _vv_candidates SDL3::SDL3-shared)
    endif ()

    foreach(_c IN LISTS _vv_candidates)
        # Use generator expressions so that configuration works for multi-config generators
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "[vulkan-visualizer] Copying runtime from target: ${_c}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:${_c}> $<TARGET_FILE_DIR:${target}>
            VERBATIM)
    endforeach()

    # Extra user-specified DLLs
    if (HINAPE_VULKAN_VISUALIZER_EXTRA_DLLS)
        separate_arguments(_vv_extra NATIVE_COMMAND "${HINAPE_VULKAN_VISUALIZER_EXTRA_DLLS}")
        foreach(_dll IN LISTS _vv_extra)
            set(_resolved "")
            if (IS_ABSOLUTE "${_dll}")
                set(_resolved "${_dll}")
            else ()
                if (EXISTS "${CMAKE_BINARY_DIR}/${_dll}")
                    set(_resolved "${CMAKE_BINARY_DIR}/${_dll}")
                elseif (EXISTS "${CMAKE_SOURCE_DIR}/${_dll}")
                    set(_resolved "${CMAKE_SOURCE_DIR}/${_dll}")
                endif ()
            endif ()
            if (_resolved AND EXISTS "${_resolved}")
                add_custom_command(TARGET ${target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E echo "[vulkan-visualizer] Copy extra DLL: ${_resolved}"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_resolved}" $<TARGET_FILE_DIR:${target}>
                    VERBATIM)
            else ()
                message(WARNING "HINAPE_VULKAN_VISUALIZER_EXTRA_DLLS entry not found: ${_dll}")
            endif ()
        endforeach()
    endif ()
endfunction()

function(use_vulkan_visualizer target)
    if (NOT TARGET ${target})
        message(FATAL_ERROR "use_vulkan_visualizer called with unknown target `${target}`")
    endif ()
    if (TARGET VulkanVisualizer::vulkan_visualizer)
        target_link_libraries(${target} PUBLIC VulkanVisualizer::vulkan_visualizer)
        target_compile_definitions(${target} PUBLIC HINAPE_HAVE_VULKAN_VISUALIZER=1)
        _vv_copy_runtime_dlls(${target})
    else ()
        message(WARNING "use_vulkan_visualizer: VulkanVisualizer::vulkan_visualizer not available; feature disabled for `${target}`")
    endif ()
endfunction()
