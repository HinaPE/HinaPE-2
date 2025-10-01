#include "cloth.h"
#include "cloth/core/arena.h"
#include "cloth/core/topology.h"
#include "cloth/model/cloth_data.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory_resource>
#include <vector>

namespace HinaPE::detail {

    using std::size_t;
    constexpr f32 k_epsilon = 1e-8f;

    struct StaticState {
        usize n_verts{0};
        std::pmr::vector<std::pmr::vector<u32>> batches;
        SolvePolicy solve{};
        ExecPolicy exec{};
        explicit StaticState(std::pmr::memory_resource* mr) : batches(mr) {}
    };

    struct DynamicState {
        model::ClothData data;
        std::pmr::vector<float> prev_x, prev_y, prev_z, lambda;
        explicit DynamicState(std::pmr::memory_resource* mr) : data(mr), prev_x(mr), prev_y(mr), prev_z(mr), lambda(mr) {}
    };

    class NativeSim final : public ISim {
    public:
        NativeSim() : mem64(64), arena(&mem64), st(&arena), dy(&arena) {}

        void init(const InitDesc& desc) {
            assert(desc.positions_xyz.size() % 3 == 0);
            const usize n = desc.positions_xyz.size() / 3;
            st.n_verts    = n;
            st.solve      = desc.solve;
            st.exec       = desc.exec;

            std::vector<std::pair<u32, u32>> edges;
            topo::build_edges_from_triangles(desc.triangles, edges);
            std::vector<std::vector<u32>> adj;
            topo::build_adjacency(n, edges, adj);
            std::vector<std::vector<u32>> batches_idx;
            topo::greedy_edge_coloring(n, edges, adj, batches_idx);
            st.batches.clear();
            st.batches.reserve(batches_idx.size());
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
                dy.data.inv_mass[i]                           = 1.0f;
            }
            for (u32 f : desc.fixed_indices)
                if (f < n) dy.data.inv_mass[f] = 0.0f;

            for (usize k = 0; k < edges.size(); ++k) {
                dy.data.e_i[k] = edges[k].first;
                dy.data.e_j[k] = edges[k].second;
            }
            for (usize k = 0; k < edges.size(); ++k) {
                u32 i = dy.data.e_i[k], j = dy.data.e_j[k];
                float dx            = dy.data.px[i] - dy.data.px[j];
                float dy_           = dy.data.py[i] - dy.data.py[j];
                float dz            = dy.data.pz[i] - dy.data.pz[j];
                dy.data.rest_len[k] = std::sqrt(dx * dx + dy_ * dy_ + dz * dz);
            }
            if (st.solve.compliance_stretch > 0.0f) dy.lambda.assign(edges.size(), 0.0f);
        }

        void step(const StepParams& params) noexcept override {
            const int sub             = std::max(1, st.solve.substeps);
            const int iters           = std::max(1, st.solve.iterations);
            const f32 full_dt         = params.dt > 0.0f ? params.dt : f32(1.0f / 60.0f);
            const f32 dt              = full_dt / static_cast<f32>(sub);
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
            v.pos_x = pv.px;
            v.pos_y = pv.py;
            v.pos_z = pv.pz;
            v.vel_x = dy.data.vx.data();
            v.vel_y = dy.data.vy.data();
            v.vel_z = dy.data.vz.data();
            v.count = pv.n;
            return v;
        }

    private:
        core::aligned_resource mem64;
        std::pmr::monotonic_buffer_resource arena;
        StaticState st;
        DynamicState dy;

        void predict_positions(const StepParams& p, f32 dt) noexcept {
            const f32 gx = p.gravity_x, gy = p.gravity_y, gz = p.gravity_z;
            const usize n = dy.data.inv_mass.size();
            for (usize i = 0; i < n; ++i) {
                const f32 im = dy.data.inv_mass[i];
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

        void solve_stretch_batch(const std::pmr::vector<u32>& batch, f32 compliance, f32 dt) noexcept {
            const f32 alpha = compliance / (dt * dt);
            for (u32 e : batch) {
                u32 i = dy.data.e_i[e], j = dy.data.e_j[e];
                f32 wi = dy.data.inv_mass[i], wj = dy.data.inv_mass[j];
                if (wi + wj == 0.0f) continue;
                f32 dx   = dy.data.px[i] - dy.data.px[j];
                f32 dy_  = dy.data.py[i] - dy.data.py[j];
                f32 dz   = dy.data.pz[i] - dy.data.pz[j];
                f32 len2 = dx * dx + dy_ * dy_ + dz * dz;
                if (len2 < k_epsilon) continue;
                f32 len     = std::sqrt(len2);
                f32 C       = len - dy.data.rest_len[e];
                f32 inv_len = 1.0f / len;
                f32 dl      = -(C + alpha * dy.lambda[e]) / (wi + wj + alpha);
                dy.lambda[e] += dl;
                f32 sx = dl * dx * inv_len, sy = dl * dy_ * inv_len, sz = dl * dz * inv_len;
                dy.data.px[i] += wi * sx;
                dy.data.py[i] += wi * sy;
                dy.data.pz[i] += wi * sz;
                dy.data.px[j] -= wj * sx;
                dy.data.py[j] -= wj * sy;
                dy.data.pz[j] -= wj * sz;
            }
        }
        void solve_stretch_batch_nc(const std::pmr::vector<u32>& batch) noexcept {
            for (u32 e : batch) {
                u32 i = dy.data.e_i[e], j = dy.data.e_j[e];
                f32 wi = dy.data.inv_mass[i], wj = dy.data.inv_mass[j];
                if (wi + wj == 0.0f) continue;
                f32 dx   = dy.data.px[i] - dy.data.px[j];
                f32 dy_  = dy.data.py[i] - dy.data.py[j];
                f32 dz   = dy.data.pz[i] - dy.data.pz[j];
                f32 len2 = dx * dx + dy_ * dy_ + dz * dz;
                if (len2 < k_epsilon) continue;
                f32 len     = std::sqrt(len2);
                f32 inv_len = 1.0f / len;
                f32 C       = len - dy.data.rest_len[e];
                f32 dl      = -C / (wi + wj);
                f32 sx = dl * dx * inv_len, sy = dl * dy_ * inv_len, sz = dl * dz * inv_len;
                dy.data.px[i] += wi * sx;
                dy.data.py[i] += wi * sy;
                dy.data.pz[i] += wi * sz;
                dy.data.px[j] -= wj * sx;
                dy.data.py[j] -= wj * sy;
                dy.data.pz[j] -= wj * sz;
            }
        }

        void integrate(f32 dt, f32 damping) noexcept {
            const usize n = dy.data.inv_mass.size();
            const f32 k   = std::clamp(1.0f - damping, 0.0f, 1.0f);
            for (usize i = 0; i < n; ++i) {
                if (dy.data.inv_mass[i] > 0.0f) {
                    f32 vx        = (dy.data.px[i] - dy.prev_x[i]) / dt;
                    f32 vy        = (dy.data.py[i] - dy.prev_y[i]) / dt;
                    f32 vz        = (dy.data.pz[i] - dy.prev_z[i]) / dt;
                    dy.data.vx[i] = vx * k;
                    dy.data.vy[i] = vy * k;
                    dy.data.vz[i] = vz * k;
                } else {
                    dy.data.vx[i] = dy.data.vy[i] = dy.data.vz[i] = 0.0f;
                    dy.data.px[i]                                 = dy.prev_x[i];
                    dy.data.py[i]                                 = dy.prev_y[i];
                    dy.data.pz[i]                                 = dy.prev_z[i];
                }
            }
        }
    };

    ISim* make_native(const InitDesc& desc) {
        auto* s = new NativeSim();
        s->init(desc);
        return s;
    }
}
