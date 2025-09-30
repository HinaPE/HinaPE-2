#include "xpbd.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <vector>

namespace HinaPE {

    namespace detail {

        using std::size_t;

        constexpr usize k_align = 64;
        constexpr f32 k_epsilon = 1e-8f;

        template <class T, usize Align>
        struct aligned_allocator {
            using value_type             = T;
            aligned_allocator() noexcept = default;
            template <class U>
            aligned_allocator(const aligned_allocator<U, Align>&) noexcept {}
            [[nodiscard]] T* allocate(std::size_t n) {
                if (n > (std::numeric_limits<std::size_t>::max)() / sizeof(T)) throw std::bad_array_new_length();
                void* p = ::operator new(n * sizeof(T), std::align_val_t(Align));
                return static_cast<T*>(p);
            }
            void deallocate(T* p, std::size_t) noexcept {
                ::operator delete(p, std::align_val_t(Align));
            }
            template <class U>
            struct rebind {
                using other = aligned_allocator<U, Align>;
            };
        };

        template <class T, usize A, class U, usize B>
        constexpr bool operator==(const aligned_allocator<T, A>&, const aligned_allocator<U, B>&) noexcept {
            return A == B;
        }

        template <class T>
        using avec = std::vector<T, aligned_allocator<T, k_align>>;

        struct bitset_dyn {
            std::vector<std::uint64_t> w;
            void ensure(int bit) {
                usize need = static_cast<usize>(bit / 64 + 1);
                if (w.size() < need) w.resize(need, 0);
            }
            void set(int bit) {
                ensure(bit);
                w[static_cast<usize>(bit / 64)] |= (std::uint64_t(1) << (bit & 63));
            }
            [[nodiscard]] bool test(int bit) const noexcept {
                usize idx = static_cast<usize>(bit / 64);
                if (idx >= w.size()) return false;
                return (w[idx] >> (bit & 63)) & 1ULL;
            }
        };

        struct StaticState {
            usize n_verts{0};
            usize n_edges{0};
            avec<u32> e_i;
            avec<u32> e_j;
            avec<f32> rest_len;
            avec<int> edge_color;
            std::vector<avec<u32>> batches;
            SolvePolicy solve;
        };

        struct DynamicState {
            avec<f32> pos_x;
            avec<f32> pos_y;
            avec<f32> pos_z;
            avec<f32> prev_x;
            avec<f32> prev_y;
            avec<f32> prev_z;
            avec<f32> vel_x;
            avec<f32> vel_y;
            avec<f32> vel_z;
            avec<f32> inv_mass;
            avec<f32> lambda_stretch; // size == n_edges iff compliance > 0
        };

        struct Sim {
            StaticState st;
            DynamicState dy;
        };

        inline void build_edges_from_triangles(const std::span<const u32> tris, avec<std::pair<u32, u32>>& edges) {
            const usize m = tris.size() / 3;
            edges.clear();
            edges.reserve(m * 3);
            for (usize t = 0; t < m; ++t) {
                u32 a   = tris[t * 3 + 0];
                u32 b   = tris[t * 3 + 1];
                u32 c   = tris[t * 3 + 2];
                u32 e0a = a < b ? a : b;
                u32 e0b = a < b ? b : a;
                u32 e1a = b < c ? b : c;
                u32 e1b = b < c ? c : b;
                u32 e2a = c < a ? c : a;
                u32 e2b = c < a ? a : c;
                edges.emplace_back(e0a, e0b);
                edges.emplace_back(e1a, e1b);
                edges.emplace_back(e2a, e2b);
            }
            std::sort(edges.begin(), edges.end());
            edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        }

        inline void compute_rest_length(std::span<const f32> xyz, usize /*n*/, const avec<std::pair<u32, u32>>& edges, avec<f32>& rest) noexcept {
            rest.resize(edges.size());
            for (usize k = 0; k < edges.size(); ++k) {
                auto [i, j] = edges[k];
                f32 dx      = xyz[i * 3 + 0] - xyz[j * 3 + 0];
                f32 dy      = xyz[i * 3 + 1] - xyz[j * 3 + 1];
                f32 dz      = xyz[i * 3 + 2] - xyz[j * 3 + 2];
                rest[k]     = std::sqrt(dx * dx + dy * dy + dz * dz);
            }
        }

        inline void build_adjacency(usize n, const avec<std::pair<u32, u32>>& edges, std::vector<avec<u32>>& adj) {
            adj.clear();
            adj.resize(n);
            for (u32 e = 0; e < static_cast<u32>(edges.size()); ++e) {
                auto [i, j] = edges[e];
                adj[i].push_back(e);
                adj[j].push_back(e);
            }
        }

        inline void greedy_edge_coloring(usize n, const avec<std::pair<u32, u32>>& edges, const std::vector<avec<u32>>& adj, avec<int>& color, std::vector<avec<u32>>& batches) {
            const usize m = edges.size();
            color.assign(m, -1);
            avec<u32> order(m);
            std::iota(order.begin(), order.end(), 0);
            std::vector<usize> deg(n);
            for (usize v = 0; v < n; ++v) deg[v] = adj[v].size();
            std::sort(order.begin(), order.end(), [&](u32 a, u32 b) {
                auto [ai, aj] = edges[a];
                auto [bi, bj] = edges[b];
                usize da      = std::max(deg[ai], deg[aj]);
                usize db      = std::max(deg[bi], deg[bj]);
                if (da != db) return da > db;
                return a < b;
            });
            std::vector<bitset_dyn> used(n);
            int max_color = -1;
            for (u32 idx : order) {
                auto [i, j] = edges[idx];
                int c       = 0;
                for (;;) {
                    bool ti = used[i].test(c);
                    bool tj = used[j].test(c);
                    if (!ti && !tj) break;
                    ++c;
                }
                used[i].set(c);
                used[j].set(c);
                color[idx] = c;
                if (c > max_color) max_color = c;
            }
            batches.clear();
            batches.resize(static_cast<usize>(max_color + 1));
            for (u32 e = 0; e < static_cast<u32>(m); ++e) batches[static_cast<usize>(color[e])].push_back(e);
        }

        inline void build_states(const InitDesc& desc, std::unique_ptr<Sim>& sim) {
            assert(desc.positions_xyz.size() % 3 == 0);
            auto s        = std::make_unique<Sim>();
            const usize n = desc.positions_xyz.size() / 3;
            avec<std::pair<u32, u32>> edges;
            build_edges_from_triangles(desc.triangles, edges);
            s->st.n_verts = n;
            s->st.n_edges = edges.size();
            s->st.e_i.resize(edges.size());
            s->st.e_j.resize(edges.size());
            for (usize k = 0; k < edges.size(); ++k) {
                s->st.e_i[k] = edges[k].first;
                s->st.e_j[k] = edges[k].second;
            }
            compute_rest_length(desc.positions_xyz, n, edges, s->st.rest_len);
            std::vector<avec<u32>> adj;
            build_adjacency(n, edges, adj);
            greedy_edge_coloring(n, edges, adj, s->st.edge_color, s->st.batches);
            s->st.solve = desc.solve;
            auto& dy    = s->dy;
            dy.pos_x.resize(n);
            dy.pos_y.resize(n);
            dy.pos_z.resize(n);
            dy.prev_x.resize(n);
            dy.prev_y.resize(n);
            dy.prev_z.resize(n);
            dy.vel_x.resize(n);
            dy.vel_y.resize(n);
            dy.vel_z.resize(n);
            dy.inv_mass.resize(n);
            for (usize i = 0; i < n; ++i) {
                dy.pos_x[i]    = desc.positions_xyz[i * 3 + 0];
                dy.pos_y[i]    = desc.positions_xyz[i * 3 + 1];
                dy.pos_z[i]    = desc.positions_xyz[i * 3 + 2];
                dy.prev_x[i]   = dy.pos_x[i];
                dy.prev_y[i]   = dy.pos_y[i];
                dy.prev_z[i]   = dy.pos_z[i];
                dy.vel_x[i]    = 0.0f;
                dy.vel_y[i]    = 0.0f;
                dy.vel_z[i]    = 0.0f;
                dy.inv_mass[i] = 1.0f;
            }
            for (u32 idx : desc.fixed_indices)
                if (idx < n) dy.inv_mass[idx] = 0.0f;
            if (s->st.solve.compliance_stretch > 0.0f) dy.lambda_stretch.resize(s->st.n_edges);
            sim = std::move(s);
        }

        inline void predict_positions(DynamicState& d, const StepParams& p, f32 dt) noexcept {
            const f32 gx = p.gravity_x, gy = p.gravity_y, gz = p.gravity_z;
            const usize n = d.pos_x.size();
            for (usize i = 0; i < n; ++i) {
                f32 im      = d.inv_mass[i];
                d.prev_x[i] = d.pos_x[i];
                d.prev_y[i] = d.pos_y[i];
                d.prev_z[i] = d.pos_z[i];
                if (im > 0.0f) { // likely movable
                    d.vel_x[i] += gx * dt;
                    d.vel_y[i] += gy * dt;
                    d.vel_z[i] += gz * dt;
                    d.pos_x[i] += d.vel_x[i] * dt;
                    d.pos_y[i] += d.vel_y[i] * dt;
                    d.pos_z[i] += d.vel_z[i] * dt;
                } else {
                    d.vel_x[i] = 0.0f;
                    d.vel_y[i] = 0.0f;
                    d.vel_z[i] = 0.0f;
                }
            }
        }

        inline void solve_stretch_batch(const StaticState& s, DynamicState& d, const avec<u32>& batch, f32 compliance, f32 dt) noexcept {
            const f32 alpha = compliance / (dt * dt);
            for (u32 e : batch) {
                u32 i  = s.e_i[e];
                u32 j  = s.e_j[e];
                f32 wi = d.inv_mass[i];
                f32 wj = d.inv_mass[j];
                if (wi + wj == 0.0f) continue;
                f32 dx   = d.pos_x[i] - d.pos_x[j];
                f32 dy   = d.pos_y[i] - d.pos_y[j];
                f32 dz   = d.pos_z[i] - d.pos_z[j];
                f32 len2 = dx * dx + dy * dy + dz * dz;
                if (len2 < k_epsilon) continue;
                f32 len     = std::sqrt(len2);
                f32 C       = len - s.rest_len[e];
                f32 inv_len = 1.0f / len;
                f32 nx      = dx * inv_len;
                f32 ny      = dy * inv_len;
                f32 nz      = dz * inv_len;
                f32 denom   = wi + wj + alpha;
                f32 dl      = -(C + alpha * d.lambda_stretch[e]) / denom;
                d.lambda_stretch[e] += dl;
                f32 sx = dl * nx;
                f32 sy = dl * ny;
                f32 sz = dl * nz;
                d.pos_x[i] += wi * sx;
                d.pos_y[i] += wi * sy;
                d.pos_z[i] += wi * sz;
                d.pos_x[j] -= wj * sx;
                d.pos_y[j] -= wj * sy;
                d.pos_z[j] -= wj * sz;
            }
        }

        inline void solve_stretch_batch_nc(const StaticState& s, DynamicState& d, const avec<u32>& batch) noexcept { // no compliance fast path
            for (u32 e : batch) {
                u32 i  = s.e_i[e];
                u32 j  = s.e_j[e];
                f32 wi = d.inv_mass[i];
                f32 wj = d.inv_mass[j];
                if (wi + wj == 0.0f) continue;
                f32 dx   = d.pos_x[i] - d.pos_x[j];
                f32 dy   = d.pos_y[i] - d.pos_y[j];
                f32 dz   = d.pos_z[i] - d.pos_z[j];
                f32 len2 = dx * dx + dy * dy + dz * dz;
                if (len2 < k_epsilon) continue;
                f32 len     = std::sqrt(len2);
                f32 inv_len = 1.0f / len; // single division path
                f32 C       = len - s.rest_len[e];
                f32 dl      = -C / (wi + wj);
                f32 nx      = dx * inv_len;
                f32 ny      = dy * inv_len;
                f32 nz      = dz * inv_len;
                f32 sx      = dl * nx;
                f32 sy      = dl * ny;
                f32 sz      = dl * nz;
                d.pos_x[i] += wi * sx;
                d.pos_y[i] += wi * sy;
                d.pos_z[i] += wi * sz;
                d.pos_x[j] -= wj * sx;
                d.pos_y[j] -= wj * sy;
                d.pos_z[j] -= wj * sz;
            }
        }

        inline void integrate(DynamicState& d, f32 dt, f32 damping) noexcept {
            const usize n = d.pos_x.size();
            const f32 k   = std::clamp(1.0f - damping, 0.0f, 1.0f);
            for (usize i = 0; i < n; ++i) {
                if (d.inv_mass[i] > 0.0f) {
                    f32 vx     = (d.pos_x[i] - d.prev_x[i]) / dt;
                    f32 vy     = (d.pos_y[i] - d.prev_y[i]) / dt;
                    f32 vz     = (d.pos_z[i] - d.prev_z[i]) / dt;
                    d.vel_x[i] = vx * k;
                    d.vel_y[i] = vy * k;
                    d.vel_z[i] = vz * k;
                } else { // fixed point: project exactly
                    d.vel_x[i] = 0.0f;
                    d.vel_y[i] = 0.0f;
                    d.vel_z[i] = 0.0f;
                    d.pos_x[i] = d.prev_x[i];
                    d.pos_y[i] = d.prev_y[i];
                    d.pos_z[i] = d.prev_z[i];
                }
            }
        }

        inline void step_native(Sim& sim, const StepParams& params) noexcept {
            auto& s             = sim.st;
            auto& d             = sim.dy;
            const int sub       = std::max(1, s.solve.substeps);
            const int iters     = std::max(1, s.solve.iterations);
            f32 full_dt         = params.dt > 0.0f ? params.dt : f32(1.0f / 60.0f);
            f32 dt              = full_dt / static_cast<f32>(sub);
            bool use_compliance = s.solve.compliance_stretch > 0.0f;
            for (int substep = 0; substep < sub; ++substep) {
                predict_positions(d, params, dt);
                if (use_compliance) std::fill(d.lambda_stretch.begin(), d.lambda_stretch.end(), 0.0f);
                for (int it = 0; it < iters; ++it) {
                    if (use_compliance) {
                        for (const auto& batch : s.batches) solve_stretch_batch(s, d, batch, s.solve.compliance_stretch, dt);
                    } else {
                        for (const auto& batch : s.batches) solve_stretch_batch_nc(s, d, batch);
                    }
                }
                integrate(d, dt, s.solve.damping);
            }
        }

    } // namespace detail

    using detail::Sim;

    [[nodiscard]] Handle create(const InitDesc& desc) {
        std::unique_ptr<detail::Sim> sim;
        detail::build_states(desc, sim);
        return sim.release();
    }

    void destroy(Handle h) noexcept {
        if (h) delete h;
    }

    void step(Handle h, const StepParams& params) noexcept {
        if (h) detail::step_native(*h, params);
    }

    DynamicView map_dynamic(Handle h) noexcept {
        DynamicView v{};
        if (!h) return v;
        auto& dy = h->dy;
        v.pos_x  = dy.pos_x.data();
        v.pos_y  = dy.pos_y.data();
        v.pos_z  = dy.pos_z.data();
        v.vel_x  = dy.vel_x.data();
        v.vel_y  = dy.vel_y.data();
        v.vel_z  = dy.vel_z.data();
        v.count  = dy.pos_x.size();
        return v;
    }

} // namespace HinaPE
