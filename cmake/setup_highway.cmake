include_guard(GLOBAL)

# Default to the latest branch of google/highway. Override by setting _HPE_HWY_TAG before include.
if (NOT DEFINED _HPE_HWY_TAG)
    set(_HPE_HWY_TAG "master")
endif ()

# Try an existing installation first.
if (NOT TARGET hwy::hwy AND NOT TARGET Highway::hwy)
    find_package(highway CONFIG QUIET)
    # Normalize any discovered targets
    set(_HWY_BASE "")
    if (TARGET hwy)
        set(_HWY_BASE hwy)
    elseif (TARGET highway)
        set(_HWY_BASE highway)
    elseif (TARGET hwy::hwy)
        get_target_property(_aliased hwy::hwy ALIASED_TARGET)
        if (_aliased)
            set(_HWY_BASE ${_aliased})
        endif ()
    endif ()
    if (_HWY_BASE)
        if (NOT TARGET hwy::hwy)
            add_library(hwy::hwy ALIAS ${_HWY_BASE})
        endif ()
        if (NOT TARGET Highway::hwy)
            add_library(Highway::hwy ALIAS ${_HWY_BASE})
        endif ()
    endif ()
endif ()

# If still not available, fetch the latest from upstream.
if (NOT TARGET Highway::hwy)
    include(FetchContent)
    # Disable optional extras when possible to keep builds lean.
    set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(highway
        GIT_REPOSITORY https://github.com/google/highway.git
        GIT_TAG        ${_HPE_HWY_TAG}
        GIT_SHALLOW    TRUE
        FIND_PACKAGE_ARGS)
    FetchContent_MakeAvailable(highway)

    # Normalize target naming to Highway::hwy and/or hwy::hwy without aliasing an alias
    set(_HWY_BASE "")
    if (TARGET hwy)
        set(_HWY_BASE hwy)
    elseif (TARGET highway)
        set(_HWY_BASE highway)
    elseif (TARGET hwy::hwy)
        get_target_property(_aliased hwy::hwy ALIASED_TARGET)
        if (_aliased)
            set(_HWY_BASE ${_aliased})
        endif ()
    endif ()
    if (_HWY_BASE)
        if (NOT TARGET hwy::hwy)
            add_library(hwy::hwy ALIAS ${_HWY_BASE})
        endif ()
        if (NOT TARGET Highway::hwy)
            add_library(Highway::hwy ALIAS ${_HWY_BASE})
        endif ()
    endif ()
endif ()

function(use_highway target)
    if (NOT TARGET ${target})
        message(FATAL_ERROR "use_highway called with unknown target `${target}`")
    endif ()

    if (TARGET Highway::hwy)
        target_link_libraries(${target} PUBLIC Highway::hwy)
        # Normalize feature define to project-wide SIMD availability flag
        target_compile_definitions(${target} PUBLIC HINAPE_HAVE_SIMD=1)
    else ()
        message(STATUS "Highway not available; proceeding without it for `${target}`")
    endif ()
endfunction()
