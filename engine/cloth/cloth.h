#ifndef HINAPE_CLOTH_H
#define HINAPE_CLOTH_H

#include <cstddef>
#include <cstdint>
#include <span>

namespace HinaPE {

    using u32   = std::uint32_t;
    using f32   = float;
    using usize = std::size_t;

    struct ExecPolicy {
        enum class Backend { Native, Simd, Tbb, Pd, Fem };
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

    namespace detail {
        struct ISim {
            virtual ~ISim() = default;
            virtual void step(const StepParams&) noexcept = 0;
            virtual DynamicView map_dynamic() noexcept    = 0;
        };


        ISim* make_native(const InitDesc& desc);
#if defined(HINAPE_HAVE_SIMD)
        ISim* make_simd(const InitDesc& desc);
#endif
#if defined(HINAPE_HAVE_TBB)
        ISim* make_tbb(const InitDesc& desc);
#endif
        // Projective Dynamics native solver
        ISim* make_pd_native(const InitDesc& desc);
        // Finite Element Method (membrane, cotangent Laplacian) native solver
        ISim* make_fem_native(const InitDesc& desc);
    } 

    using Handle = detail::ISim*;
    [[nodiscard]] Handle create(const InitDesc& desc);
    void destroy(Handle h) noexcept;
    void step(Handle h, const StepParams& params) noexcept;
    [[nodiscard]] DynamicView map_dynamic(Handle h) noexcept;

} 

#endif 
