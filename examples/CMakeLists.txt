cmake_minimum_required(VERSION 3.26)

set(EXAMPLE_NAME xpbd_visualizer)
set(SHADER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shader)
set(SHADER_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders_bin)
set(COMPILED_SHADER_DIR ${SHADER_SRC_DIR})
set(SHADERS ${SHADER_SRC_DIR}/cloth.vert ${SHADER_SRC_DIR}/cloth.frag)
find_program(GLSLC glslc HINTS ENV VULKAN_SDK PATH_SUFFIXES Bin bin)
if (GLSLC)
    file(MAKE_DIRECTORY ${SHADER_OUT_DIR})
    set(COMPILED_SHADER_DIR ${SHADER_OUT_DIR})
    set(SPIRV_OUTPUTS)
    foreach(SRC ${SHADERS})
        get_filename_component(FN ${SRC} NAME)
        set(OUT ${SHADER_OUT_DIR}/${FN}.spv)
        add_custom_command(
            OUTPUT ${OUT}
            COMMAND ${GLSLC} -O -c ${SRC} -o ${OUT}
            DEPENDS ${SRC}
            VERBATIM)
        list(APPEND SPIRV_OUTPUTS ${OUT})
    endforeach()
    add_custom_target(${EXAMPLE_NAME}_shaders DEPENDS ${SPIRV_OUTPUTS})
endif ()

add_executable(${EXAMPLE_NAME} xpbd_visualizer.cpp)
if (GLSLC)
    add_dependencies(${EXAMPLE_NAME} ${EXAMPLE_NAME}_shaders)
endif ()
target_compile_definitions(${EXAMPLE_NAME} PRIVATE
    SHADER_OUTPUT_DIR="${COMPILED_SHADER_DIR}"
    SHADER_SOURCE_DIR="${SHADER_SRC_DIR}")

target_link_libraries(${EXAMPLE_NAME} PRIVATE HinaPE)
include(../cmake/setup_tbb.cmake OPTIONAL)
if (COMMAND use_tbb)
    use_tbb(${EXAMPLE_NAME})
endif ()
include(../cmake/setup_vulkan_visualizer.cmake OPTIONAL)
if (COMMAND use_vulkan_visualizer)
    use_vulkan_visualizer(${EXAMPLE_NAME})
endif ()

set_target_properties(${EXAMPLE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

