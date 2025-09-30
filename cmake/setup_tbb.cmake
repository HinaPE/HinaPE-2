include_guard(GLOBAL)

# Auto-detect or fetch oneTBB without exposing options
set(_HPE_TBB_VERSION "2022.2.0")

if (NOT TARGET TBB::tbb)
    find_package(TBB CONFIG QUIET)
endif ()

if (NOT TARGET TBB::tbb)
    include(FetchContent)
    set(TBB_TEST OFF CACHE BOOL "" FORCE)
    set(TBB_STRICT OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(oneTBB
        URL https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v${_HPE_TBB_VERSION}.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP OFF)
    FetchContent_MakeAvailable(oneTBB)
    if (NOT TARGET TBB::tbb AND TARGET tbb)
        add_library(TBB::tbb ALIAS tbb)
    endif ()
endif ()

# Helper to link TBB if available
function(use_tbb target)
    if (NOT TARGET ${target})
        message(FATAL_ERROR "use_tbb called with unknown target `${target}`")
    endif ()

    if (TARGET TBB::tbb)
        target_link_libraries(${target} PUBLIC TBB::tbb)
        if (WIN32)
            get_target_property(_target_type ${target} TYPE)
            if (_target_type STREQUAL "EXECUTABLE" OR _target_type STREQUAL "SHARED_LIBRARY" OR _target_type STREQUAL "MODULE_LIBRARY")
                add_custom_command(TARGET ${target} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
                    COMMAND_EXPAND_LISTS)
            endif ()
        endif ()
    else ()
        message(STATUS "TBB unavailable; proceeding without parallel backend")
    endif ()
endfunction()
