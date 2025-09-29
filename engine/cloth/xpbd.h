#ifndef HINAPE_XPBD_H
#define HINAPE_XPBD_H
#include <cstddef>
#include <cstdint>
#include <span>


namespace HinaPE {


    using u32   = std::uint32_t;
    using f32   = float;
    using usize = std::size_t;


    struct ExecPolicy {
        enum class Backend { Native };
        Backend backend{Backend::Native};
        int threads{0};
        bool deterministic{true};
        bool telemetry{false};
    };


    struct SolvePolicy {
        int substeps{1};
        int iterations{8};
        f32 compliance_stretch{0.0f};
        f32 damping{0.01f};
    };


    struct InitDesc {
        std::span<const f32> positions_xyz;
        std::span<const u32> triangles;
        std::span<const u32> fixed_indices;
        ExecPolicy exec{};
        SolvePolicy solve{};
    };


    struct StepParams {
        f32 dt{1.0f / 60.0f};
        f32 gravity_x{0.0f};
        f32 gravity_y{-9.81f};
        f32 gravity_z{0.0f};
    };


    struct DynamicView {
        f32* pos_x{};
        f32* pos_y{};
        f32* pos_z{};
        f32* vel_x{};
        f32* vel_y{};
        f32* vel_z{};
        usize count{};
    };


    using Handle = void*;


    Handle create(const InitDesc& desc);
    void destroy(Handle h);
    void step(Handle h, const StepParams& params);
    DynamicView map_dynamic(Handle h);


} // namespace HinaPE
#endif // HINAPE_XPBD_H
