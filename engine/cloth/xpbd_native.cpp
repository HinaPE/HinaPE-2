// Native XPBD implementation (decoupled)

#include "cloth.h"



#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory_resource>
#include <new>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace HinaPE {
    namespace detail {

        using std::size_t;
        constexpr usize k_align   = 64;
        constexpr f32   k_epsilon = 1e-8f;

        using detail::aligned_resource;

        template <class T>
        using pvec = std::pmr::vector<T>;

        

        struct StaticState {
            usize n_verts{0};
            usize n_edges{0};
            pvec<u32> e_i;
            pvec<u32> e_j;
            pvec<f32> rest_len;
            std::pmr::vector<pvec<u32>> batches;
            SolvePolicy solve{};
            ExecPolicy exec{};

            explicit StaticState(std::pmr::memory_resource* mr) : e_i(mr), e_j(mr), rest_len(mr), batches(mr) {}
        };
        struct DynamicState {
            pvec<f32> pos_x, pos_y, pos_z;
            pvec<f32> prev_x, prev_y, prev_z;
            pvec<f32> vel_x, vel_y, vel_z;
            pvec<f32> inv_mass;
            pvec<f32> lambda_stretch; // optional
            explicit DynamicState(std::pmr::memory_resource* mr)
                : pos_x(mr), pos_y(mr), pos_z(mr), prev_x(mr), prev_y(mr), prev_z(mr), vel_x(mr), vel_y(mr), vel_z(mr), inv_mass(mr), lambda_stretch(mr) {}
        };

        class NativeSim final : public ISim {
        public:
            NativeSim() : mem64(k_align), arena(&mem64), st(&arena), dy(&arena) {}

            void init(const InitDesc& desc) {
                assert(desc.positions_xyz.size() % 3 == 0);
                const usize n = desc.positions_xyz.size() / 3;
                std::vector<std::pair<u32, u32>> edges;
                topo::build_edges_from_triangles(desc.triangles, edges);
                st.n_verts = n;
                st.n_edges = edges.size();
                st.e_i.resize(edges.size());
                st.e_j.resize(edges.size());
                for (usize k = 0; k < edges.size(); ++k) {
                    st.e_i[k] = edges[k].first;
                    st.e_j[k] = edges[k].second;
                }
                compute_rest_length(desc.positions_xyz, edges, st.rest_len);
                std::vector<std::vector<u32>> adj;
                topo::build_adjacency(n, edges, adj);
                std::vector<std::vector<u32>> batches_idx;
                topo::greedy_edge_coloring(n, edges, adj, batches_idx);
                st.batches.clear(); st.batches.reserve(batches_idx.size());
                auto inner_alloc = std::pmr::polymorphic_allocator<u32>(&arena);
                for (const auto& b : batches_idx) {
                    pvec<u32> out(inner_alloc);
                    out.insert(out.end(), b.begin(), b.end());
                    st.batches.push_back(std::move(out));
                }
                st.solve = desc.solve;
                st.exec  = desc.exec;

                const usize n_pad = (n + 7u) & ~usize(7);
                dy.pos_x.resize(n_pad);
                dy.pos_y.resize(n_pad);
                dy.pos_z.resize(n_pad);
                dy.prev_x.resize(n_pad);
                dy.prev_y.resize(n_pad);
                dy.prev_z.resize(n_pad);
                dy.vel_x.resize(n_pad);
                dy.vel_y.resize(n_pad);
                dy.vel_z.resize(n_pad);
                dy.inv_mass.resize(n_pad);
                for (usize i = 0; i < n; ++i) {
                    dy.pos_x[i]  = desc.positions_xyz[i * 3 + 0];
                    dy.pos_y[i]  = desc.positions_xyz[i * 3 + 1];
                    dy.pos_z[i]  = desc.positions_xyz[i * 3 + 2];
                    dy.prev_x[i] = dy.pos_x[i];
                    dy.prev_y[i] = dy.pos_y[i];
                    dy.prev_z[i] = dy.pos_z[i];
                    dy.vel_x[i] = dy.vel_y[i] = dy.vel_z[i] = 0.0f;
                    dy.inv_mass[i]                          = 1.0f;
                }
                for (usize i = n; i < n_pad; ++i) {
                    dy.pos_x[i] = dy.pos_y[i] = dy.pos_z[i] = 0.0f;
                    dy.prev_x[i] = dy.prev_y[i] = dy.prev_z[i] = 0.0f;
                    dy.vel_x[i] = dy.vel_y[i] = dy.vel_z[i] = 0.0f;
                    dy.inv_mass[i] = 0.0f;
                }
                for (u32 idx : desc.fixed_indices) if (idx < n) dy.inv_mass[idx] = 0.0f;
                if (st.solve.compliance_stretch > 0.0f) {
                    dy.lambda_stretch.resize(st.n_edges);
                    std::fill(dy.lambda_stretch.begin(), dy.lambda_stretch.end(), 0.0f);
                }
            }

            void step(const StepParams& params) noexcept override {
                int sub             = std::max(1, st.solve.substeps);
                int iters           = std::max(1, st.solve.iterations);
                f32 full_dt         = params.dt > 0.0f ? params.dt : f32(1.0f / 60.0f);
                f32 dt              = full_dt / static_cast<f32>(sub);
                bool use_compliance = st.solve.compliance_stretch > 0.0f;
                for (int ss = 0; ss < sub; ++ss) {
                    predict_positions(params, dt);
                    if (use_compliance) std::fill(dy.lambda_stretch.begin(), dy.lambda_stretch.end(), 0.0f);
                    for (int it = 0; it < iters; ++it) {
                        if (use_compliance) {
                            for (const auto& batch : st.batches) solve_stretch_batch(batch, st.solve.compliance_stretch, dt);
                        } else {
                            for (const auto& batch : st.batches) solve_stretch_batch_nc(batch);
                        }
                    }
                    integrate(dt, st.solve.damping);
                }
            }

            DynamicView map_dynamic() noexcept override {
                DynamicView v{};
                v.pos_x = dy.pos_x.data();
                v.pos_y = dy.pos_y.data();
                v.pos_z = dy.pos_z.data();
                v.vel_x = dy.vel_x.data();
                v.vel_y = dy.vel_y.data();
                v.vel_z = dy.vel_z.data();
                v.count = st.n_verts;
                return v;
            }

        private:
            aligned_resource mem64;
            std::pmr::monotonic_buffer_resource arena;
            StaticState st;
            DynamicState dy;

            

            static void compute_rest_length(std::span<const f32> xyz, const std::vector<std::pair<u32, u32>>& edges, pvec<f32>& rest) noexcept {
                rest.resize(edges.size());
                for (usize k = 0; k < edges.size(); ++k) {
                    auto [i, j] = edges[k];
                    f32 dx      = xyz[i * 3 + 0] - xyz[j * 3 + 0];
                    f32 dy      = xyz[i * 3 + 1] - xyz[j * 3 + 1];
                    f32 dz      = xyz[i * 3 + 2] - xyz[j * 3 + 2];
                    rest[k]     = std::sqrt(dx * dx + dy * dy + dz * dz);
                }
            }

            

            void predict_positions(const StepParams& p, f32 dt) noexcept {
                const f32 gx = p.gravity_x, gy = p.gravity_y, gz = p.gravity_z;
                const usize n = dy.inv_mass.size();
                for (usize i = 0; i < n; ++i) {
                    f32 im      = dy.inv_mass[i];
                    dy.prev_x[i] = dy.pos_x[i];
                    dy.prev_y[i] = dy.pos_y[i];
                    dy.prev_z[i] = dy.pos_z[i];
                    if (im > 0.0f) {
                        dy.vel_x[i] += gx * dt; dy.vel_y[i] += gy * dt; dy.vel_z[i] += gz * dt;
                        dy.pos_x[i] += dy.vel_x[i] * dt; dy.pos_y[i] += dy.vel_y[i] * dt; dy.pos_z[i] += dy.vel_z[i] * dt;
                    } else {
                        dy.vel_x[i] = dy.vel_y[i] = dy.vel_z[i] = 0.0f;
                    }
                }
            }

            void solve_stretch_batch(const pvec<u32>& batch, f32 compliance, f32 dt) noexcept {
                const f32 alpha = compliance / (dt * dt);
                for (u32 e : batch) {
                    u32 i  = st.e_i[e]; u32 j = st.e_j[e];
                    f32 wi = dy.inv_mass[i]; f32 wj = dy.inv_mass[j];
                    if (wi + wj == 0.0f) continue;
                    f32 dx = dy.pos_x[i] - dy.pos_x[j]; f32 dy_ = dy.pos_y[i] - dy.pos_y[j]; f32 dz = dy.pos_z[i] - dy.pos_z[j];
                    f32 len2 = dx * dx + dy_ * dy_ + dz * dz; if (len2 < k_epsilon) continue;
                    f32 len = std::sqrt(len2); f32 C = len - st.rest_len[e]; f32 inv_len = 1.0f / len;
                    f32 dl = -(C + alpha * dy.lambda_stretch[e]) / (wi + wj + alpha); dy.lambda_stretch[e] += dl;
                    f32 sx = dl * dx * inv_len; f32 sy = dl * dy_ * inv_len; f32 sz = dl * dz * inv_len;
                    dy.pos_x[i] += wi * sx; dy.pos_y[i] += wi * sy; dy.pos_z[i] += wi * sz;
                    dy.pos_x[j] -= wj * sx; dy.pos_y[j] -= wj * sy; dy.pos_z[j] -= wj * sz;
                }
            }
            void solve_stretch_batch_nc(const pvec<u32>& batch) noexcept {
                for (u32 e : batch) {
                    u32 i  = st.e_i[e]; u32 j = st.e_j[e];
                    f32 wi = dy.inv_mass[i]; f32 wj = dy.inv_mass[j];
                    if (wi + wj == 0.0f) continue;
                    f32 dx = dy.pos_x[i] - dy.pos_x[j]; f32 dy_ = dy.pos_y[i] - dy.pos_y[j]; f32 dz = dy.pos_z[i] - dy.pos_z[j];
                    f32 len2 = dx * dx + dy_ * dy_ + dz * dz; if (len2 < k_epsilon) continue;
                    f32 len = std::sqrt(len2); f32 inv_len = 1.0f / len; f32 C = len - st.rest_len[e]; f32 dl = -C / (wi + wj);
                    f32 sx = dl * dx * inv_len; f32 sy = dl * dy_ * inv_len; f32 sz = dl * dz * inv_len;
                    dy.pos_x[i] += wi * sx; dy.pos_y[i] += wi * sy; dy.pos_z[i] += wi * sz;
                    dy.pos_x[j] -= wj * sx; dy.pos_y[j] -= wj * sy; dy.pos_z[j] -= wj * sz;
                }
            }

            void integrate(f32 dt, f32 damping) noexcept {
                const usize n = dy.inv_mass.size();
                const f32 k = std::clamp(1.0f - damping, 0.0f, 1.0f);
                for (usize i = 0; i < n; ++i) {
                    if (dy.inv_mass[i] > 0.0f) {
                        f32 vx = (dy.pos_x[i] - dy.prev_x[i]) / dt;
                        f32 vy = (dy.pos_y[i] - dy.prev_y[i]) / dt;
                        f32 vz = (dy.pos_z[i] - dy.prev_z[i]) / dt;
                        dy.vel_x[i] = vx * k; dy.vel_y[i] = vy * k; dy.vel_z[i] = vz * k;
                    } else {
                        dy.vel_x[i] = dy.vel_y[i] = dy.vel_z[i] = 0.0f;
                        dy.pos_x[i] = dy.prev_x[i]; dy.pos_y[i] = dy.prev_y[i]; dy.pos_z[i] = dy.prev_z[i];
                    }
                }
            }
        };

        ISim* make_native(const InitDesc& desc) {
            auto* s = new NativeSim();
            s->init(desc);
            return s;
        }

    } // namespace detail
} // namespace HinaPE



