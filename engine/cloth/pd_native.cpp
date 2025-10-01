#include "cloth.h"
#include "cloth/core/arena.h"
#include "cloth/core/topology.h"
#include "cloth/model/cloth_data.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory_resource>
#include <numeric>
#include <vector>

namespace HinaPE {
namespace detail {

using std::size_t;

// Simple, high‑performance Projective Dynamics solver for stretch constraints
// on a triangle cloth mesh. This implementation uses a matrix‑free Preconditioned
// Conjugate Gradient (PCG) solve for the global step with a Jacobi preconditioner.

struct PDStaticState {
    usize n_verts{0};
    usize n_edges{0};
    std::pmr::vector<u32> deg;             // degree per vertex (for Laplacian diag)
    SolvePolicy solve{};
    ExecPolicy exec{};
    explicit PDStaticState(std::pmr::memory_resource* mr) : deg(mr) {}
};

struct PDDynamicState {
    model::ClothData data;                  // positions, velocities, inv_mass, edges, rest_len
    std::pmr::vector<float> prev_x, prev_y, prev_z; // previous positions for velocity update
    // Local step projected edge vectors (p_e)
    std::pmr::vector<float> pex, pey, pez;
    // PCG temporaries and diagonals (reused each substep/iteration to avoid allocations)
    std::pmr::vector<float> rhsx, rhsy, rhsz;
    std::pmr::vector<float> x, r, z, p, Ap; // scratch for a single scalar solve; reused per‑axis
    std::pmr::vector<float> diagK;          // Jacobi preconditioner (diag of K)

    explicit PDDynamicState(std::pmr::memory_resource* mr)
        : data(mr), prev_x(mr), prev_y(mr), prev_z(mr), pex(mr), pey(mr), pez(mr),
          rhsx(mr), rhsy(mr), rhsz(mr), x(mr), r(mr), z(mr), p(mr), Ap(mr), diagK(mr) {}
};

class PDNativeSim final : public ISim {
public:
    PDNativeSim() : mem64(64), arena(&mem64), st(&arena), dy(&arena) {}

    void init(const InitDesc& desc) {
        assert(desc.positions_xyz.size() % 3 == 0);
        const usize n = desc.positions_xyz.size() / 3;
        st.n_verts     = n;
        st.solve       = desc.solve;
        st.exec        = desc.exec;

        // Build edge list from triangles
        std::vector<std::pair<u32,u32>> edges_vec; topo::build_edges_from_triangles(desc.triangles, edges_vec);
        st.n_edges = edges_vec.size();

        // Prepare data containers
        dy.data.resize_particles(n, true);
        dy.prev_x.resize(n); dy.prev_y.resize(n); dy.prev_z.resize(n);
        dy.data.resize_edges(st.n_edges);
        st.deg.assign(n, 0);
        dy.pex.resize(st.n_edges); dy.pey.resize(st.n_edges); dy.pez.resize(st.n_edges);
        dy.rhsx.resize(n); dy.rhsy.resize(n); dy.rhsz.resize(n);
        dy.x.resize(n); dy.r.resize(n); dy.z.resize(n); dy.p.resize(n); dy.Ap.resize(n);
        dy.diagK.resize(n);

        // Initialize particle state
        for (usize i = 0; i < n; ++i) {
            dy.data.px[i] = desc.positions_xyz[i*3+0];
            dy.data.py[i] = desc.positions_xyz[i*3+1];
            dy.data.pz[i] = desc.positions_xyz[i*3+2];
            dy.prev_x[i]  = dy.data.px[i];
            dy.prev_y[i]  = dy.data.py[i];
            dy.prev_z[i]  = dy.data.pz[i];
            dy.data.vx[i] = dy.data.vy[i] = dy.data.vz[i] = 0.0f;
            dy.data.inv_mass[i] = 1.0f; // unit mass by default; fixed vertices set below
        }
        for (u32 f : desc.fixed_indices) if (f < n) dy.data.inv_mass[f] = 0.0f;

        // Copy edges + rest lengths, and compute degree per vertex
        for (usize k = 0; k < st.n_edges; ++k) {
            dy.data.e_i[k] = edges_vec[k].first;
            dy.data.e_j[k] = edges_vec[k].second;
            ++st.deg[dy.data.e_i[k]]; ++st.deg[dy.data.e_j[k]];
        }
        for (usize k = 0; k < st.n_edges; ++k) {
            const u32 i = dy.data.e_i[k], j = dy.data.e_j[k];
            const float dx = dy.data.px[i] - dy.data.px[j];
            const float dy_ = dy.data.py[i] - dy.data.py[j];
            const float dz = dy.data.pz[i] - dy.data.pz[j];
            dy.data.rest_len[k] = std::sqrt(dx*dx + dy_*dy_ + dz*dz);
        }
    }

    void step(const StepParams& params) noexcept override {
        const int sub   = std::max(1, st.solve.substeps);
        const int iters = std::max(1, st.solve.iterations); // used as PCG iteration budget
        const float full_dt = params.dt > 0.0f ? params.dt : float(1.0/60.0);
        const float dt = full_dt / static_cast<float>(sub);

        // Map compliance to a stiffness weight. If compliance==0, use a large stiffness.
        const float stiffness = st.solve.compliance_stretch > 0.0f
                              ? (1.0f / std::max(st.solve.compliance_stretch, 1e-8f))
                              : 1.0e4f; // robust high stiffness default

        for (int s = 0; s < sub; ++s) {
            predict_positions_(params, dt);
            local_project_edges_();
            build_rhs_and_diagonal_(stiffness);
            // Global solve: three independent scalar solves sharing the same K
            pcg_solve_axis_(dy.rhsx, dy.data.px, stiffness, iters);
            pcg_solve_axis_(dy.rhsy, dy.data.py, stiffness, iters);
            pcg_solve_axis_(dy.rhsz, dy.data.pz, stiffness, iters);
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
    PDStaticState st;
    PDDynamicState dy;

    static constexpr float kEps = 1e-8f;

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

    void local_project_edges_() noexcept {
        // Compute per‑edge projections p_e = rest_len * normalize(x_i - x_j)
        const usize m = st.n_edges;
        for (usize e = 0; e < m; ++e) {
            const u32 i = dy.data.e_i[e], j = dy.data.e_j[e];
            float dx = dy.data.px[i] - dy.data.px[j];
            float dy_ = dy.data.py[i] - dy.data.py[j];
            float dz = dy.data.pz[i] - dy.data.pz[j];
            float len2 = dx*dx + dy_*dy_ + dz*dz;
            if (len2 < kEps) { dy.pex[e] = dy.pey[e] = dy.pez[e] = 0.0f; continue; }
            float inv_len = 1.0f / std::sqrt(len2);
            float scale = dy.data.rest_len[e] * inv_len;
            dy.pex[e] = dx * scale;
            dy.pey[e] = dy_ * scale;
            dy.pez[e] = dz * scale;
        }
    }

    void build_rhs_and_diagonal_(float k) noexcept {
        const usize n = st.n_verts; const usize m = st.n_edges;
        // Compute diagonal for Jacobi preconditioner: diag(K) = M + k * diag(L)
        for (usize i = 0; i < n; ++i) {
            const float im = dy.data.inv_mass[i];
            const bool fixed = im == 0.0f;
            // Use large mass for fixed vertices to approximate Dirichlet conditions in PCG
            const float mass = fixed ? 1.0e9f : 1.0f; // unit mass for free vertices
            dy.diagK[i] = mass + k * static_cast<float>(st.deg[i]);
        }
        // Build RHS: b = M*y + k * A^T p  (per axis)
        std::fill(dy.rhsx.begin(), dy.rhsx.end(), 0.0f);
        std::fill(dy.rhsy.begin(), dy.rhsy.end(), 0.0f);
        std::fill(dy.rhsz.begin(), dy.rhsz.end(), 0.0f);
        for (usize i = 0; i < n; ++i) {
            const float im = dy.data.inv_mass[i]; const bool fixed = im == 0.0f;
            const float mass = fixed ? 1.0e9f : 1.0f;
            dy.rhsx[i] = mass * dy.data.px[i];
            dy.rhsy[i] = mass * dy.data.py[i];
            dy.rhsz[i] = mass * dy.data.pz[i];
        }
        for (usize e = 0; e < m; ++e) {
            const u32 i = dy.data.e_i[e], j = dy.data.e_j[e];
            const float px = dy.pex[e], py = dy.pey[e], pz = dy.pez[e];
            const float wk = k; // constant weight per edge
            dy.rhsx[i] += wk * px; dy.rhsy[i] += wk * py; dy.rhsz[i] += wk * pz;
            dy.rhsx[j] -= wk * px; dy.rhsy[j] -= wk * py; dy.rhsz[j] -= wk * pz;
        }
    }

    // Matrix‑free product y = K * x where K = M + k * L and L is the (weighted) graph Laplacian.
    void apply_K_(const float* in, float* out, float k) const noexcept {
        const usize n = st.n_verts; const usize m = st.n_edges;
        // out = M * in
        for (usize i = 0; i < n; ++i) {
            const float im = dy.data.inv_mass[i]; const bool fixed = im == 0.0f;
            const float mass = fixed ? 1.0e9f : 1.0f;
            out[i] = mass * in[i];
        }
        // out += k * L * in
        for (usize e = 0; e < m; ++e) {
            const u32 i = dy.data.e_i[e], j = dy.data.e_j[e];
            const float diff = in[i] - in[j];
            const float wk = k;
            out[i] += wk * diff;
            out[j] -= wk * diff;
        }
    }

    void pcg_solve_axis_(const std::pmr::vector<float>& b, std::pmr::vector<float>& x_out, float k, int maxIters) noexcept {
        const usize n = st.n_verts;
        // Initial guess: current predicted positions already in x_out
        auto& x = dy.x; x.assign(x_out.begin(), x_out.begin() + n);
        auto& r = dy.r; auto& z = dy.z; auto& p = dy.p; auto& Ap = dy.Ap; auto& D = dy.diagK;

        // r = b - K x
        apply_K_(x.data(), Ap.data(), k);
        for (usize i = 0; i < n; ++i) r[i] = b[i] - Ap[i];
        // z = D^{-1} r (Jacobi preconditioner)
        for (usize i = 0; i < n; ++i) z[i] = r[i] / std::max(D[i], 1e-12f);
        p = z;
        float rz_old = std::inner_product(r.begin(), r.begin() + n, z.begin(), 0.0f);
        const float tol = 1e-4f;
        for (int it = 0; it < maxIters; ++it) {
            apply_K_(p.data(), Ap.data(), k);
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
        // Write back solution
        for (usize i = 0; i < n; ++i) x_out[i] = x[i];
    }

    void integrate_(float dt, float damping) noexcept {
        const usize n = dy.data.inv_mass.size();
        const float k = std::clamp(1.0f - damping, 0.0f, 1.0f);
        for (usize i = 0; i < n; ++i) {
            if (dy.data.inv_mass[i] > 0.0f) {
                const float vx = (dy.data.px[i] - dy.prev_x[i]) / dt;
                const float vy = (dy.data.py[i] - dy.prev_y[i]) / dt;
                const float vz = (dy.data.pz[i] - dy.prev_z[i]) / dt;
                dy.data.vx[i] = vx * k; dy.data.vy[i] = vy * k; dy.data.vz[i] = vz * k;
            } else {
                dy.data.vx[i] = dy.data.vy[i] = dy.data.vz[i] = 0.0f;
                dy.data.px[i] = dy.prev_x[i]; dy.data.py[i] = dy.prev_y[i]; dy.data.pz[i] = dy.prev_z[i];
            }
        }
    }
};

// Optional factory (not wired into the public API yet). Downstream code may
// declare it manually to instantiate a PD solver.
ISim* make_pd_native(const InitDesc& desc) {
    auto* s = new PDNativeSim();
    s->init(desc);
    return s;
}

} // namespace detail
} // namespace HinaPE

