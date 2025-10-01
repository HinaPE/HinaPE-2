#include "cloth.h"
#include "cloth/core/arena.h"
#include "cloth/core/topology.h"
#include "cloth/model/cloth_data.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory_resource>
#include <numeric>
#include <utility>
#include <vector>

namespace HinaPE {
namespace detail {

using std::size_t;

// High‑performance finite element membrane solver using a cotangent Laplacian
// stiffness matrix and implicit Euler time integration. Matrix‑free PCG keeps
// memory and CPU overhead low while remaining stable for large time steps.

struct FEMStaticState {
    usize n_verts{0};
    std::pmr::vector<u32> e_i, e_j;     // undirected edge list
    std::pmr::vector<float> w_e;        // cotangent weight per edge
    std::pmr::vector<float> deg_w;      // weighted degree per vertex (sum of adjacent weights)
    SolvePolicy solve{};
    ExecPolicy exec{};
    explicit FEMStaticState(std::pmr::memory_resource* mr)
        : e_i(mr), e_j(mr), w_e(mr), deg_w(mr) {}
};

struct FEMDynamicState {
    model::ClothData data;              // x, v, inv_mass
    std::pmr::vector<float> prev_x, prev_y, prev_z; // previous x for velocity update
    // PCG scratch
    std::pmr::vector<float> x, r, z, p, Ap, rhs, diagA, mass;
    explicit FEMDynamicState(std::pmr::memory_resource* mr)
        : data(mr), prev_x(mr), prev_y(mr), prev_z(mr),
          x(mr), r(mr), z(mr), p(mr), Ap(mr), rhs(mr), diagA(mr), mass(mr) {}
};

static inline float safe_cot(const float ax, const float ay, const float az,
                             const float bx, const float by, const float bz) noexcept {
    // cot(angle between a and b) = dot(a,b) / |a x b|
    const float dot = ax * bx + ay * by + az * bz;
    const float cx  = ay * bz - az * by;
    const float cy  = az * bx - ax * bz;
    const float cz  = ax * by - ay * bx;
    const float n   = std::sqrt(cx * cx + cy * cy + cz * cz);
    if (n < 1e-12f) return 0.0f;
    return dot / n;
}

class FEMNativeSim final : public ISim {
public:
    FEMNativeSim() : mem64(64), arena(&mem64), st(&arena), dy(&arena) {}

    void init(const InitDesc& desc) {
        assert(desc.positions_xyz.size() % 3 == 0);
        const usize n = desc.positions_xyz.size() / 3;
        st.n_verts     = n;
        st.solve       = desc.solve;
        st.exec        = desc.exec;

        // Initialize dynamic arrays
        dy.data.resize_particles(n, true);
        dy.prev_x.resize(n); dy.prev_y.resize(n); dy.prev_z.resize(n);
        dy.mass.resize(n);

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
        for (usize i = 0; i < n; ++i) {
            dy.mass[i] = (dy.data.inv_mass[i] > 0.0f) ? (1.0f / dy.data.inv_mass[i]) : 1.0e9f;
        }

        // Build cotangent weights from triangles
        build_cotangent_edges_(desc.triangles, desc.positions_xyz);

        // PCG buffers
        dy.x.resize(n); dy.r.resize(n); dy.z.resize(n); dy.p.resize(n); dy.Ap.resize(n); dy.rhs.resize(n); dy.diagA.resize(n);
    }

    void step(const StepParams& params) noexcept override {
        const int sub = std::max(1, st.solve.substeps);
        const float full_dt = params.dt > 0.0f ? params.dt : float(1.0/60.0);
        const float dt = full_dt / static_cast<float>(sub);
        // Compliance maps to stiffness; default to high stiffness when compliance is zero
        const float k = st.solve.compliance_stretch > 0.0f ? (1.0f / std::max(st.solve.compliance_stretch, 1e-8f)) : 1.0e4f;
        const float dt2k = dt * dt * k;

        for (int s = 0; s < sub; ++s) {
            predict_positions_(params, dt);
            build_system_(dt2k);
            // Solve A x = b separately for x/y/z using same A (matrix‑free)
            pcg_solve_axis_(dy.data.px);
            pcg_solve_axis_(dy.data.py);
            pcg_solve_axis_(dy.data.pz);
            integrate_(dt, st.solve.damping);
        }
    }

    DynamicView map_dynamic() noexcept override {
        DynamicView v{}; auto pv = dy.data.particles();
        v.pos_x = pv.px; v.pos_y = pv.py; v.pos_z = pv.pz;
        v.vel_x = dy.data.vx.data(); v.vel_y = dy.data.vy.data(); v.vel_z = dy.data.vz.data();
        v.count = pv.n; return v;
    }

private:
    core::aligned_resource mem64;
    std::pmr::monotonic_buffer_resource arena;
    FEMStaticState st;
    FEMDynamicState dy;

    void build_cotangent_edges_(std::span<const u32> tris, std::span<const float> xyz) {
        const usize m = tris.size() / 3;
        std::pmr::vector<std::pair<std::pair<u32,u32>, float>> accum{&arena};
        accum.reserve(m * 3);
        auto add_weight = [&](u32 a, u32 b, float w) {
            if (a > b) std::swap(a, b);
            accum.emplace_back(std::make_pair(a, b), w);
        };
        for (usize t = 0; t < m; ++t) {
            u32 i = tris[t * 3 + 0], j = tris[t * 3 + 1], k = tris[t * 3 + 2];
            const float ix = xyz[i*3+0], iy = xyz[i*3+1], iz = xyz[i*3+2];
            const float jx = xyz[j*3+0], jy = xyz[j*3+1], jz = xyz[j*3+2];
            const float kx = xyz[k*3+0], ky = xyz[k*3+1], kz = xyz[k*3+2];
            // Vectors opposite each angle
            float v0x = jx - ix, v0y = jy - iy, v0z = jz - iz; // i->j
            float w0x = kx - ix, w0y = ky - iy, w0z = kz - iz; // i->k
            float v1x = kx - jx, v1y = ky - jy, v1z = kz - jz; // j->k
            float w1x = ix - jx, w1y = iy - jy, w1z = iz - jz; // j->i
            float v2x = ix - kx, v2y = iy - ky, v2z = iz - kz; // k->i
            float w2x = jx - kx, w2y = jy - ky, w2z = jz - kz; // k->j
            float cot_i = safe_cot(v0x, v0y, v0z, w0x, w0y, w0z);
            float cot_j = safe_cot(v1x, v1y, v1z, w1x, w1y, w1z);
            float cot_k = safe_cot(v2x, v2y, v2z, w2x, w2y, w2z);
            // Each edge gets half of the two opposing cots
            add_weight(j, k, 0.5f * cot_i); // opposite i
            add_weight(k, i, 0.5f * cot_j); // opposite j
            add_weight(i, j, 0.5f * cot_k); // opposite k
        }
        std::sort(accum.begin(), accum.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
        st.e_i.clear(); st.e_j.clear(); st.w_e.clear(); st.deg_w.assign(st.n_verts, 0.0f);
        for (usize idx = 0; idx < accum.size();) {
            const auto key = accum[idx].first;
            float w = 0.0f; do { w += accum[idx].second; ++idx; } while (idx < accum.size() && accum[idx].first == key);
            if (w <= 0.0f) continue; // skip degenerate
            st.e_i.push_back(key.first); st.e_j.push_back(key.second); st.w_e.push_back(w);
            st.deg_w[key.first]  += w;
            st.deg_w[key.second] += w;
        }
    }

    void predict_positions_(const StepParams& p, float dt) noexcept {
        const float gx = p.gravity_x, gy = p.gravity_y, gz = p.gravity_z;
        const usize n = dy.data.inv_mass.size();
        for (usize i = 0; i < n; ++i) {
            const float im = dy.data.inv_mass[i];
            dy.prev_x[i] = dy.data.px[i];
            dy.prev_y[i] = dy.data.py[i];
            dy.prev_z[i] = dy.data.pz[i];
            if (im > 0.0f) {
                dy.data.vx[i] += gx * dt;
                dy.data.vy[i] += gy * dt;
                dy.data.vz[i] += gz * dt;
                dy.data.px[i] += dy.data.vx[i] * dt;
                dy.data.py[i] += dy.data.vy[i] * dt;
                dy.data.pz[i] += dy.data.vz[i] * dt;
            } else {
                dy.data.vx[i] = dy.data.vy[i] = dy.data.vz[i] = 0.0f;
            }
        }
    }

    void build_system_(float dt2k) noexcept {
        const usize n = st.n_verts;
        // Diagonal preconditioner: A = M + dt^2 k L  => diag(A) = mass + dt^2 k * deg_w
        for (usize i = 0; i < n; ++i) dy.diagA[i] = dy.mass[i] + dt2k * st.deg_w[i];
        // RHS b = M * y (current predicted positions already in dy.data.p{xyz})
        // We reuse dy.rhs as a scalar buffer per axis during PCG, filled in pcg_solve_axis_
        // to avoid extra copies.
        // Store dt2k for matvec
        dt2k_ = dt2k;
    }

    void apply_A_(const float* in, float* out) const noexcept {
        const usize n = st.n_verts; const usize m = st.e_i.size();
        // out = M * in
        for (usize i = 0; i < n; ++i) out[i] = dy.mass[i] * in[i];
        // out += dt^2 k * L * in
        const float s = dt2k_;
        for (usize e = 0; e < m; ++e) {
            const u32 a = st.e_i[e], b = st.e_j[e]; const float w = st.w_e[e];
            const float d = in[a] - in[b];
            const float c = s * w;
            out[a] += c * d;
            out[b] -= c * d;
        }
    }

    void pcg_solve_axis_(std::pmr::vector<float>& axis) noexcept {
        const usize n = st.n_verts;
        auto& x = dy.x; x.assign(axis.begin(), axis.begin() + n);
        auto& r = dy.r; auto& z = dy.z; auto& p = dy.p; auto& Ap = dy.Ap; auto& D = dy.diagA; auto& b = dy.rhs;
        // b = M * y_axis
        for (usize i = 0; i < n; ++i) b[i] = dy.mass[i] * axis[i];
        // r = b - A x
        apply_A_(x.data(), Ap.data());
        for (usize i = 0; i < n; ++i) r[i] = b[i] - Ap[i];
        for (usize i = 0; i < n; ++i) z[i] = r[i] / std::max(D[i], 1e-12f);
        p = z;
        float rz_old = std::inner_product(r.begin(), r.begin() + n, z.begin(), 0.0f);
        const float tol = 1e-4f;
        const int maxIters = std::max(1, st.solve.iterations);
        for (int it = 0; it < maxIters; ++it) {
            apply_A_(p.data(), Ap.data());
            float pAp = std::inner_product(p.begin(), p.begin() + n, Ap.begin(), 0.0f);
            if (std::fabs(pAp) < 1e-20f) break;
            float alpha = rz_old / pAp;
            for (usize i = 0; i < n; ++i) x[i] += alpha * p[i];
            for (usize i = 0; i < n; ++i) r[i] -= alpha * Ap[i];
            float rnorm2 = std::inner_product(r.begin(), r.begin() + n, r.begin(), 0.0f);
            if (rnorm2 <= tol * tol) break;
            for (usize i = 0; i < n; ++i) z[i] = r[i] / std::max(D[i], 1e-12f);
            float rz_new = std::inner_product(r.begin(), r.begin() + n, z.begin(), 0.0f);
            float beta = (rz_new / std::max(rz_old, 1e-30f));
            for (usize i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
            rz_old = rz_new;
        }
        for (usize i = 0; i < n; ++i) axis[i] = x[i];
    }

    void integrate_(float dt, float damping) noexcept {
        const usize n = dy.data.inv_mass.size();
        const float k = std::clamp(1.0f - damping, 0.0f, 1.0f);
        for (usize i = 0; i < n; ++i) {
            if (dy.data.inv_mass[i] > 0.0f) {
                float vx = (dy.data.px[i] - dy.prev_x[i]) / dt;
                float vy = (dy.data.py[i] - dy.prev_y[i]) / dt;
                float vz = (dy.data.pz[i] - dy.prev_z[i]) / dt;
                dy.data.vx[i] = vx * k; dy.data.vy[i] = vy * k; dy.data.vz[i] = vz * k;
            } else {
                dy.data.vx[i] = dy.data.vy[i] = dy.data.vz[i] = 0.0f;
                dy.data.px[i] = dy.prev_x[i]; dy.data.py[i] = dy.prev_y[i]; dy.data.pz[i] = dy.prev_z[i];
            }
        }
    }

    float dt2k_{0.0f};
};

ISim* make_fem_native(const InitDesc& desc) {
    auto* s = new FEMNativeSim();
    s->init(desc);
    return s;
}

} // namespace detail
} // namespace HinaPE

