file(GLOB_RECURSE hpe_src *.h *.cpp)

set(lib_name HinaPE)
add_library(${lib_name} ${hpe_src})
target_include_directories(${lib_name}
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/cloth
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
if (HINAPE_WARNINGS)
    target_compile_options(${lib_name} PRIVATE ${HINAPE_WARNINGS})
endif ()
if (HINAPE_RELEASE_OPTS)
    target_compile_options(${lib_name} PRIVATE ${HINAPE_RELEASE_OPTS})
endif ()
if (HINAPE_PUBLIC_DEFINES)
    target_compile_definitions(${lib_name} PUBLIC ${HINAPE_PUBLIC_DEFINES})
endif ()
if (HINAPE_PRIVATE_DEFINES)
    target_compile_definitions(${lib_name} PRIVATE ${HINAPE_PRIVATE_DEFINES})
endif ()
include(../cmake/setup_tbb.cmake OPTIONAL)
if (TARGET TBB::tbb)
    use_tbb(${lib_name})
    target_compile_definitions(${lib_name} PUBLIC HINAPE_HAVE_TBB=1)
endif ()

if (HINAPE_HAVE_AVX2 AND HINAPE_AVX2_FLAGS)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/cloth/xpbd_avx2.cpp
        PROPERTIES COMPILE_OPTIONS "${HINAPE_AVX2_FLAGS}")
    target_compile_definitions(${lib_name} PUBLIC HINAPE_HAVE_AVX2=1)
endif ()
