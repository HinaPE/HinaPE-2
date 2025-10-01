// Highway SIMD XPBD implementation
#include "cloth.h"
#include "cloth/core/arena.h"
#include "cloth/core/topology.h"
#include "cloth/model/cloth_data.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory_resource>
#include <numeric>
#include <vector>

#if defined(HINAPE_HAVE_SIMD)
#include <hwy/highway.h>
#endif

// Highway kernels live in the global hwy namespace (not inside HinaPE)
#if defined(HINAPE_HAVE_SIMD)
namespace hwy {
namespace HWY_NAMESPACE {

using Df   = HWY_FULL(float);
using Du32 = Rebind<uint32_t, Df>;
using Dsi  = Rebind<int32_t, Df>;
constexpr float kHwyEps = 1e-8f;

// Predict positions: update prev_*; v += g*dt; p += v*dt for im>0, else v=0
static void HwyPredictPositions(const float* HWY_RESTRICT gravity, // gx,gy,gz
                                float* HWY_RESTRICT px,
                                float* HWY_RESTRICT py,
                                float* HWY_RESTRICT pz,
                                float* HWY_RESTRICT vx,
                                float* HWY_RESTRICT vy,
                                float* HWY_RESTRICT vz,
                                const float* HWY_RESTRICT inv_mass,
                                float* HWY_RESTRICT prev_x,
                                float* HWY_RESTRICT prev_y,
                                float* HWY_RESTRICT prev_z,
                                float dt,
                                size_t n) {
  const Df d;
  const auto gx = Set(d, gravity[0] * dt);
  const auto gy = Set(d, gravity[1] * dt);
  const auto gz = Set(d, gravity[2] * dt);
  const auto vdt = Set(d, dt);
  const auto zero = Zero(d);

  for (size_t i = 0; i < n; i += Lanes(d)) {
    const size_t remaining = n - i;
    const auto m = FirstN(d, remaining);

    const auto px0 = MaskedLoad(m, d, px + i);
    const auto py0 = MaskedLoad(m, d, py + i);
    const auto pz0 = MaskedLoad(m, d, pz + i);
    BlendedStore(px0, m, d, prev_x + i);
    BlendedStore(py0, m, d, prev_y + i);
    BlendedStore(pz0, m, d, prev_z + i);

    const auto im = MaskedLoad(m, d, inv_mass + i);
    const auto movable = Gt(im, zero);

    auto vx0 = MaskedLoad(m, d, vx + i);
    auto vy0 = MaskedLoad(m, d, vy + i);
    auto vz0 = MaskedLoad(m, d, vz + i);

    auto vx1 = vx0 + gx;
    auto vy1 = vy0 + gy;
    auto vz1 = vz0 + gz;

    // p += v*dt
    auto px1 = MulAdd(vx1, vdt, px0);
    auto py1 = MulAdd(vy1, vdt, py0);
    auto pz1 = MulAdd(vz1, vdt, pz0);

    // If fixed: keep p, set v=0
    auto vx_out = IfThenElse(movable, vx1, zero);
    auto vy_out = IfThenElse(movable, vy1, zero);
    auto vz_out = IfThenElse(movable, vz1, zero);
    auto px_out = IfThenElse(movable, px1, px0);
    auto py_out = IfThenElse(movable, py1, py0);
    auto pz_out = IfThenElse(movable, pz1, pz0);

    BlendedStore(px_out, m, d, px + i);
    BlendedStore(py_out, m, d, py + i);
    BlendedStore(pz_out, m, d, pz + i);
    BlendedStore(vx_out, m, d, vx + i);
    BlendedStore(vy_out, m, d, vy + i);
    BlendedStore(vz_out, m, d, vz + i);
  }
}

// Integrate velocities and apply damping; restore fixed positions
static void HwyIntegrate(float* HWY_RESTRICT px,
                         float* HWY_RESTRICT py,
                         float* HWY_RESTRICT pz,
                         float* HWY_RESTRICT vx,
                         float* HWY_RESTRICT vy,
                         float* HWY_RESTRICT vz,
                         const float* HWY_RESTRICT inv_mass,
                         const float* HWY_RESTRICT prev_x,
                         const float* HWY_RESTRICT prev_y,
                         const float* HWY_RESTRICT prev_z,
                         float dt,
                         float damping,
                         size_t n) {
  const Df d;
  const auto zero = Zero(d);
  const float k = std::clamp(1.0f - damping, 0.0f, 1.0f);
  const auto vk = Set(d, k);
  const auto inv_dt = Set(d, 1.0f / dt);

  for (size_t i = 0; i < n; i += Lanes(d)) {
    const size_t remaining = n - i;
    const auto m = FirstN(d, remaining);

    auto px0 = MaskedLoad(m, d, px + i);
    auto py0 = MaskedLoad(m, d, py + i);
    auto pz0 = MaskedLoad(m, d, pz + i);
    const auto pxp = MaskedLoad(m, d, prev_x + i);
    const auto pyp = MaskedLoad(m, d, prev_y + i);
    const auto pzp = MaskedLoad(m, d, prev_z + i);
    const auto im  = MaskedLoad(m, d, inv_mass + i);
    const auto movable = Gt(im, zero);

    const auto vx_new = Mul((px0 - pxp), inv_dt) * vk;
    const auto vy_new = Mul((py0 - pyp), inv_dt) * vk;
    const auto vz_new = Mul((pz0 - pzp), inv_dt) * vk;

    // Fixed: zero velocity, restore positions
    const auto vx_out = IfThenElse(movable, vx_new, zero);
    const auto vy_out = IfThenElse(movable, vy_new, zero);
    const auto vz_out = IfThenElse(movable, vz_new, zero);
    px0 = IfThenElse(movable, px0, pxp);
    py0 = IfThenElse(movable, py0, pyp);
    pz0 = IfThenElse(movable, pz0, pzp);

    BlendedStore(px0, m, d, px + i);
    BlendedStore(py0, m, d, py + i);
    BlendedStore(pz0, m, d, pz + i);
    BlendedStore(vx_out, m, d, vx + i);
    BlendedStore(vy_out, m, d, vy + i);
    BlendedStore(vz_out, m, d, vz + i);
  }
}

// Solve distance constraints without compliance
static void HwySolveStretchNC(const uint32_t* HWY_RESTRICT batch,
                              size_t batch_size,
                              const uint32_t* HWY_RESTRICT e_i,
                              const uint32_t* HWY_RESTRICT e_j,
                              const float* HWY_RESTRICT rest,
                              float* HWY_RESTRICT px,
                              float* HWY_RESTRICT py,
                              float* HWY_RESTRICT pz,
                              const float* HWY_RESTRICT inv_mass) {
  const Df d;
  const Du32 du;
  const Dsi dsi;
  const auto zero = Zero(d);
  const auto eps  = Set(d, kHwyEps);

  for (size_t k = 0; k < batch_size; k += Lanes(d)) {
    const size_t rem = batch_size - k;
    const auto mu   = FirstN(du, rem);
    const auto ve_u = MaskedLoad(mu, du, batch + k); // edge indices (u32)
    const auto ve   = BitCast(dsi, ve_u);            // to i32

    const auto vi = GatherIndex(dsi, reinterpret_cast<const int32_t*>(e_i), ve);
    const auto vj = GatherIndex(dsi, reinterpret_cast<const int32_t*>(e_j), ve);

    // Gather particle data
    const auto pxi = GatherIndex(d, px, vi);
    const auto pyi = GatherIndex(d, py, vi);
    const auto pzi = GatherIndex(d, pz, vi);
    const auto pxj = GatherIndex(d, px, vj);
    const auto pyj = GatherIndex(d, py, vj);
    const auto pzj = GatherIndex(d, pz, vj);

    const auto wi = GatherIndex(d, inv_mass, vi);
    const auto wj = GatherIndex(d, inv_mass, vj);
    const auto wsum = wi + wj;
    const auto movable = Gt(wsum, zero);

    const auto dx = pxi - pxj;
    const auto dy = pyi - pyj;
    const auto dz = pzi - pzj;
    const auto len2 = MulAdd(dx, dx, MulAdd(dy, dy, dz * dz));
    const auto ok = And(Gt(len2, eps), movable);

    const auto len = Sqrt(len2);
    const auto inv_len = Set(d, 1.0f) / len;
    const auto C = len - GatherIndex(d, rest, ve);
    const auto dl = Neg(C) / wsum;
    const auto sx = dl * dx * inv_len;
    const auto sy = dl * dy * inv_len;
    const auto sz = dl * dz * inv_len;

    const auto pxi2 = IfThenElse(ok, MulAdd(wi, sx, pxi), pxi);
    const auto pyi2 = IfThenElse(ok, MulAdd(wi, sy, pyi), pyi);
    const auto pzi2 = IfThenElse(ok, MulAdd(wi, sz, pzi), pzi);
    const auto pxj2 = IfThenElse(ok, pxj - wj * sx, pxj);
    const auto pyj2 = IfThenElse(ok, pyj - wj * sy, pyj);
    const auto pzj2 = IfThenElse(ok, pzj - wj * sz, pzj);

    MaskedScatterIndex(pxi2, ok, d, px, vi);
    MaskedScatterIndex(pyi2, ok, d, py, vi);
    MaskedScatterIndex(pzi2, ok, d, pz, vi);
    MaskedScatterIndex(pxj2, ok, d, px, vj);
    MaskedScatterIndex(pyj2, ok, d, py, vj);
    MaskedScatterIndex(pzj2, ok, d, pz, vj);
  }
}

// Solve distance constraints with XPBD compliance
static void HwySolveStretchXPBD(const uint32_t* HWY_RESTRICT batch,
                                size_t batch_size,
                                const uint32_t* HWY_RESTRICT e_i,
                                const uint32_t* HWY_RESTRICT e_j,
                                const float* HWY_RESTRICT rest,
                                float* HWY_RESTRICT px,
                                float* HWY_RESTRICT py,
                                float* HWY_RESTRICT pz,
                                const float* HWY_RESTRICT inv_mass,
                                float* HWY_RESTRICT lambda,
                                float alpha) {
  const Df d;
  const Du32 du;
  const Dsi dsi;
  const auto zero = Zero(d);
  const auto eps  = Set(d, kHwyEps);
  const auto valpha = Set(d, alpha);

  for (size_t k = 0; k < batch_size; k += Lanes(d)) {
    const size_t rem = batch_size - k;
    const auto mu   = FirstN(du, rem);
    const auto ve_u = MaskedLoad(mu, du, batch + k);
    const auto ve   = BitCast(dsi, ve_u);

    const auto vi = GatherIndex(dsi, reinterpret_cast<const int32_t*>(e_i), ve);
    const auto vj = GatherIndex(dsi, reinterpret_cast<const int32_t*>(e_j), ve);

    const auto pxi = GatherIndex(d, px, vi);
    const auto pyi = GatherIndex(d, py, vi);
    const auto pzi = GatherIndex(d, pz, vi);
    const auto pxj = GatherIndex(d, px, vj);
    const auto pyj = GatherIndex(d, py, vj);
    const auto pzj = GatherIndex(d, pz, vj);

    const auto wi = GatherIndex(d, inv_mass, vi);
    const auto wj = GatherIndex(d, inv_mass, vj);
    const auto wsum = wi + wj;
    const auto movable = Gt(wsum, zero);

    const auto dx = pxi - pxj;
    const auto dy = pyi - pyj;
    const auto dz = pzi - pzj;
    const auto len2 = MulAdd(dx, dx, MulAdd(dy, dy, dz * dz));
    const auto ok = And(Gt(len2, eps), movable);

    const auto len = Sqrt(len2);
    const auto inv_len = Set(d, 1.0f) / len;
    const auto C = len - GatherIndex(d, rest, ve);
    const auto lam = GatherIndex(d, lambda, ve);
    const auto dl = Neg(C + valpha * lam) / (wsum + valpha);
    const auto lam2 = lam + dl;
    MaskedScatterIndex(lam2, ok, d, lambda, ve);

    const auto sx = dl * dx * inv_len;
    const auto sy = dl * dy * inv_len;
    const auto sz = dl * dz * inv_len;

    const auto pxi2 = IfThenElse(ok, MulAdd(wi, sx, pxi), pxi);
    const auto pyi2 = IfThenElse(ok, MulAdd(wi, sy, pyi), pyi);
    const auto pzi2 = IfThenElse(ok, MulAdd(wi, sz, pzi), pzi);
    const auto pxj2 = IfThenElse(ok, pxj - wj * sx, pxj);
    const auto pyj2 = IfThenElse(ok, pyj - wj * sy, pyj);
    const auto pzj2 = IfThenElse(ok, pzj - wj * sz, pzj);

    MaskedScatterIndex(pxi2, ok, d, px, vi);
    MaskedScatterIndex(pyi2, ok, d, py, vi);
    MaskedScatterIndex(pzi2, ok, d, pz, vi);
    MaskedScatterIndex(pxj2, ok, d, px, vj);
    MaskedScatterIndex(pyj2, ok, d, py, vj);
    MaskedScatterIndex(pzj2, ok, d, pz, vj);
  }
}

}  // namespace HWY_NAMESPACE
}


#endif // HINAPE_HAVE_SIMD

namespace HinaPE {
namespace detail {
using std::size_t;

struct SimdStatic {
    usize n_verts{0};
    std::pmr::vector<std::pmr::vector<u32>> batches;
    SolvePolicy solve{};
    ExecPolicy exec{};
    explicit SimdStatic(std::pmr::memory_resource* mr) : batches(mr) {}
};

struct SimdDynamic {
    model::ClothData data;
    std::pmr::vector<float> prev_x, prev_y, prev_z, lambda;
    explicit SimdDynamic(std::pmr::memory_resource* mr)
        : data(mr), prev_x(mr), prev_y(mr), prev_z(mr), lambda(mr) {}
};

/* duplicate Highway block removed */

class SimdSim final : public ISim {
public:
    SimdSim() : mem64(64), arena(&mem64), st(&arena), dy(&arena) {}

    void init(const InitDesc& desc) {
        assert(desc.positions_xyz.size() % 3 == 0);
        const usize n = desc.positions_xyz.size() / 3;
        st.n_verts     = n;
        st.solve       = desc.solve;
        st.exec        = desc.exec;

        std::vector<std::pair<u32, u32>> edges; topo::build_edges_from_triangles(desc.triangles, edges);
        std::vector<std::vector<u32>> adj;   topo::build_adjacency(n, edges, adj);
        std::vector<std::vector<u32>> batches_idx; topo::greedy_edge_coloring(n, edges, adj, batches_idx);
        st.batches.clear(); st.batches.reserve(batches_idx.size());
        for (const auto& b : batches_idx) {
            std::pmr::vector<u32> out{std::pmr::polymorphic_allocator<u32>(&arena)};
            out.assign(b.begin(), b.end());
            st.batches.push_back(std::move(out));
        }

        dy.data.resize_particles(n, true);
        dy.prev_x.resize(dy.data.px.size());
        dy.prev_y.resize(dy.data.py.size());
        dy.prev_z.resize(dy.data.pz.size());
        dy.data.resize_edges(edges.size());

        for (usize i = 0; i < n; ++i) {
            dy.data.px[i] = desc.positions_xyz[i * 3 + 0];
            dy.data.py[i] = desc.positions_xyz[i * 3 + 1];
            dy.data.pz[i] = desc.positions_xyz[i * 3 + 2];
            dy.prev_x[i]  = dy.data.px[i];
            dy.prev_y[i]  = dy.data.py[i];
            dy.prev_z[i]  = dy.data.pz[i];
            dy.data.vx[i] = dy.data.vy[i] = dy.data.vz[i] = 0.0f;
            dy.data.inv_mass[i] = 1.0f;
        }
        for (u32 f : desc.fixed_indices) if (f < n) dy.data.inv_mass[f] = 0.0f;

        for (usize k = 0; k < edges.size(); ++k) { dy.data.e_i[k] = edges[k].first; dy.data.e_j[k] = edges[k].second; }
        for (usize k = 0; k < edges.size(); ++k) {
            u32 i = dy.data.e_i[k], j = dy.data.e_j[k];
            float dx = dy.data.px[i] - dy.data.px[j];
            float dy_ = dy.data.py[i] - dy.data.py[j];
            float dz = dy.data.pz[i] - dy.data.pz[j];
            dy.data.rest_len[k] = std::sqrt(dx * dx + dy_ * dy_ + dz * dz);
        }
        if (st.solve.compliance_stretch > 0.0f) dy.lambda.assign(edges.size(), 0.0f);
    }

    void step(const StepParams& params) noexcept override {
        const int sub   = std::max(1, st.solve.substeps);
        const int iters = std::max(1, st.solve.iterations);
        const f32 full_dt = params.dt > 0.0f ? params.dt : f32(1.0f / 60.0f);
        const f32 dt = full_dt / static_cast<f32>(sub);
        const bool use_compliance = st.solve.compliance_stretch > 0.0f;
        for (int s = 0; s < sub; ++s) {
            predict_positions(params, dt);
            if (use_compliance) std::fill(dy.lambda.begin(), dy.lambda.end(), 0.0f);
            for (int it = 0; it < iters; ++it) {
                if (use_compliance) {
                    for (const auto& b : st.batches) solve_stretch_batch(b, st.solve.compliance_stretch, dt);
                } else {
                    for (const auto& b : st.batches) solve_stretch_batch_nc(b);
                }
            }
            integrate(dt, st.solve.damping);
        }
    }

    DynamicView map_dynamic() noexcept override {
        DynamicView v{};
        auto pv = dy.data.particles();
        v.pos_x = pv.px; v.pos_y = pv.py; v.pos_z = pv.pz;
        v.vel_x = dy.data.vx.data(); v.vel_y = dy.data.vy.data(); v.vel_z = dy.data.vz.data();
        v.count = pv.n;
        return v;
    }

private:
    core::aligned_resource mem64;
    std::pmr::monotonic_buffer_resource arena;
    SimdStatic st;
    SimdDynamic dy;

    void predict_positions(const StepParams& p, f32 dt) noexcept {
        const usize n = dy.data.inv_mass.size();
#if defined(HINAPE_HAVE_SIMD)
        float g[3] = {p.gravity_x, p.gravity_y, p.gravity_z};
        hwy::HWY_NAMESPACE::HwyPredictPositions(g,
            dy.data.px.data(), dy.data.py.data(), dy.data.pz.data(),
            dy.data.vx.data(), dy.data.vy.data(), dy.data.vz.data(),
            dy.data.inv_mass.data(),
            dy.prev_x.data(), dy.prev_y.data(), dy.prev_z.data(),
            dt, n);
#else
        (void)p; (void)dt; (void)n;
#endif
    }

    void solve_stretch_batch(const std::pmr::vector<u32>& batch, f32 compliance, f32 dt) noexcept {
#if defined(HINAPE_HAVE_SIMD)
        if (batch.empty()) return;
        const float alpha = compliance / (dt * dt);
        hwy::HWY_NAMESPACE::HwySolveStretchXPBD(
            batch.data(), batch.size(),
            dy.data.e_i.data(), dy.data.e_j.data(), dy.data.rest_len.data(),
            dy.data.px.data(), dy.data.py.data(), dy.data.pz.data(),
            dy.data.inv_mass.data(), dy.lambda.data(), alpha);
#else
        (void)batch; (void)compliance; (void)dt;
#endif
    }

    void solve_stretch_batch_nc(const std::pmr::vector<u32>& batch) noexcept {
#if defined(HINAPE_HAVE_SIMD)
        if (batch.empty()) return;
        hwy::HWY_NAMESPACE::HwySolveStretchNC(
            batch.data(), batch.size(),
            dy.data.e_i.data(), dy.data.e_j.data(), dy.data.rest_len.data(),
            dy.data.px.data(), dy.data.py.data(), dy.data.pz.data(),
            dy.data.inv_mass.data());
#else
        (void)batch;
#endif
    }

    void integrate(f32 dt, f32 damping) noexcept {
        const usize n = dy.data.inv_mass.size();
#if defined(HINAPE_HAVE_SIMD)
        hwy::HWY_NAMESPACE::HwyIntegrate(
            dy.data.px.data(), dy.data.py.data(), dy.data.pz.data(),
            dy.data.vx.data(), dy.data.vy.data(), dy.data.vz.data(),
            dy.data.inv_mass.data(),
            dy.prev_x.data(), dy.prev_y.data(), dy.prev_z.data(),
            dt, damping, n);
#else
        (void)dt; (void)damping; (void)n;
#endif
    }
};

#if defined(HINAPE_HAVE_SIMD)
ISim* make_simd(const InitDesc& desc) {
    auto* s = new SimdSim();
    s->init(desc);
    return s;
}
#else
ISim* make_simd(const InitDesc& desc) {
    return make_native(desc);
}
#endif

} // namespace detail
} // namespace HinaPE
