#include "xpbd.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <vector>
#ifdef HINAPE_HAVE_AVX2
#include <immintrin.h>
#endif

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
            ExecPolicy exec;
        };
        struct DynamicState {
            avec<f32> pos_x, pos_y, pos_z, prev_x, prev_y, prev_z, vel_x, vel_y, vel_z, inv_mass, lambda_stretch;
        };
        struct Sim {
            StaticState st;
            DynamicState dy;
        };

        inline void build_edges_from_triangles(std::span<const u32> tris, avec<std::pair<u32, u32>>& edges) {
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
        inline void compute_rest_length(std::span<const f32> xyz, const avec<std::pair<u32, u32>>& edges, avec<f32>& rest) noexcept {
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
            adj.assign(n, {});
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
            int maxc = -1;
            for (u32 idx : order) {
                auto [i, j] = edges[idx];
                int c       = 0;
                for (;;) {
                    if (!used[i].test(c) && !used[j].test(c)) break;
                    ++c;
                }
                used[i].set(c);
                used[j].set(c);
                color[idx] = c;
                if (c > maxc) maxc = c;
            }
            batches.assign(static_cast<usize>(maxc + 1), {});
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
            compute_rest_length(desc.positions_xyz, edges, s->st.rest_len);
            std::vector<avec<u32>> adj;
            build_adjacency(n, edges, adj);
            greedy_edge_coloring(n, edges, adj, s->st.edge_color, s->st.batches);
            s->st.solve = desc.solve;
            s->st.exec  = desc.exec;
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
                dy.pos_x[i]  = desc.positions_xyz[i * 3 + 0];
                dy.pos_y[i]  = desc.positions_xyz[i * 3 + 1];
                dy.pos_z[i]  = desc.positions_xyz[i * 3 + 2];
                dy.prev_x[i] = dy.pos_x[i];
                dy.prev_y[i] = dy.pos_y[i];
                dy.prev_z[i] = dy.pos_z[i];
                dy.vel_x[i] = dy.vel_y[i] = dy.vel_z[i] = 0.0f;
                dy.inv_mass[i]                          = 1.0f;
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
                if (im > 0.0f) {
                    d.vel_x[i] += gx * dt;
                    d.vel_y[i] += gy * dt;
                    d.vel_z[i] += gz * dt;
                    d.pos_x[i] += d.vel_x[i] * dt;
                    d.pos_y[i] += d.vel_y[i] * dt;
                    d.pos_z[i] += d.vel_z[i] * dt;
                } else {
                    d.vel_x[i] = d.vel_y[i] = d.vel_z[i] = 0.0f;
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
                f32 dl      = -(C + alpha * d.lambda_stretch[e]) / (wi + wj + alpha);
                d.lambda_stretch[e] += dl;
                f32 sx = dl * dx * inv_len;
                f32 sy = dl * dy * inv_len;
                f32 sz = dl * dz * inv_len;
                d.pos_x[i] += wi * sx;
                d.pos_y[i] += wi * sy;
                d.pos_z[i] += wi * sz;
                d.pos_x[j] -= wj * sx;
                d.pos_y[j] -= wj * sy;
                d.pos_z[j] -= wj * sz;
            }
        }
        inline void solve_stretch_batch_nc(const StaticState& s, DynamicState& d, const avec<u32>& batch) noexcept {
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
                f32 inv_len = 1.0f / len;
                f32 C       = len - s.rest_len[e];
                f32 dl      = -C / (wi + wj);
                f32 sx      = dl * dx * inv_len;
                f32 sy      = dl * dy * inv_len;
                f32 sz      = dl * dz * inv_len;
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
                } else {
                    d.vel_x[i] = d.vel_y[i] = d.vel_z[i] = 0.0f;
                    d.pos_x[i]                           = d.prev_x[i];
                    d.pos_y[i]                           = d.prev_y[i];
                    d.pos_z[i]                           = d.prev_z[i];
                }
            }
        }
        inline void step_native(Sim& sim, const StepParams& params) noexcept {
            auto& s             = sim.st;
            auto& d             = sim.dy;
            int sub             = std::max(1, s.solve.substeps);
            int iters           = std::max(1, s.solve.iterations);
            f32 full_dt         = params.dt > 0.0f ? params.dt : f32(1.0f / 60.0f);
            f32 dt              = full_dt / static_cast<f32>(sub);
            bool use_compliance = s.solve.compliance_stretch > 0.0f;
            for (int ss = 0; ss < sub; ++ss) {
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

#ifdef HINAPE_HAVE_AVX2
        // AVX2 helpers
        inline void solve_stretch_batch_avx2(const StaticState& s, DynamicState& d, const avec<u32>& batch, f32 compliance, f32 dt) noexcept {
            if (batch.empty()) return;
            const f32 alpha      = compliance / (dt * dt);
            const __m256 v_alpha = _mm256_set1_ps(alpha);
            const __m256 v_eps   = _mm256_set1_ps(k_epsilon);
            for (usize off = 0; off < batch.size(); off += 8) {
                alignas(32) u32 e_idx[8]{};
                int lane_count = (int) std::min<usize>(8, batch.size() - off);
                for (int lane = 0; lane < lane_count; ++lane) e_idx[lane] = batch[off + lane];
                alignas(32) int idx_i[8]{};
                alignas(32) int idx_j[8]{};
                for (int lane = 0; lane < lane_count; ++lane) {
                    u32 e       = e_idx[lane];
                    idx_i[lane] = (int) s.e_i[e];
                    idx_j[lane] = (int) s.e_j[e];
                }
                __m256i vi        = _mm256_load_si256(reinterpret_cast<const __m256i*>(idx_i));
                __m256i vj        = _mm256_load_si256(reinterpret_cast<const __m256i*>(idx_j));
                __m256 xi         = _mm256_i32gather_ps(d.pos_x.data(), vi, 4);
                __m256 yi         = _mm256_i32gather_ps(d.pos_y.data(), vi, 4);
                __m256 zi         = _mm256_i32gather_ps(d.pos_z.data(), vi, 4);
                __m256 xj         = _mm256_i32gather_ps(d.pos_x.data(), vj, 4);
                __m256 yj         = _mm256_i32gather_ps(d.pos_y.data(), vj, 4);
                __m256 zj         = _mm256_i32gather_ps(d.pos_z.data(), vj, 4);
                __m256 dx         = _mm256_sub_ps(xi, xj);
                __m256 dy         = _mm256_sub_ps(yi, yj);
                __m256 dz         = _mm256_sub_ps(zi, zj);
                __m256 len2       = _mm256_fmadd_ps(dz, dz, _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dx, dx)));
                __m256 mask_valid = _mm256_cmp_ps(len2, v_eps, _CMP_GT_OQ);
                __m256 wi         = _mm256_i32gather_ps(d.inv_mass.data(), vi, 4);
                __m256 wj         = _mm256_i32gather_ps(d.inv_mass.data(), vj, 4);
                __m256 wsum       = _mm256_add_ps(wi, wj);
                __m256 mask_mass  = _mm256_cmp_ps(wsum, _mm256_set1_ps(0.0f), _CMP_GT_OQ);
                __m256 vmask      = _mm256_and_ps(mask_valid, mask_mass);
                if (_mm256_testz_ps(vmask, vmask)) continue;
                __m256 len = _mm256_sqrt_ps(len2);
                alignas(32) int e_int[8]{};
                for (int lane = 0; lane < lane_count; ++lane) e_int[lane] = (int) e_idx[lane];
                __m256i ve        = _mm256_load_si256(reinterpret_cast<const __m256i*>(e_int));
                __m256 rest       = _mm256_i32gather_ps(s.rest_len.data(), ve, 4);
                __m256 C          = _mm256_sub_ps(len, rest);
                __m256 lambda_old = _mm256_i32gather_ps(d.lambda_stretch.data(), ve, 4);
                __m256 denom      = _mm256_add_ps(wsum, v_alpha);
                __m256 tmp        = _mm256_fmadd_ps(v_alpha, lambda_old, C);
                __m256 dl         = _mm256_div_ps(_mm256_sub_ps(_mm256_setzero_ps(), tmp), denom);
                __m256 lambda_new = _mm256_add_ps(lambda_old, dl);
                __m256 inv_len    = _mm256_div_ps(_mm256_set1_ps(1.0f), len);
                __m256 nx         = _mm256_mul_ps(dx, inv_len);
                __m256 ny         = _mm256_mul_ps(dy, inv_len);
                __m256 nz         = _mm256_mul_ps(dz, inv_len);
                __m256 sx         = _mm256_mul_ps(dl, nx);
                __m256 sy         = _mm256_mul_ps(dl, ny);
                __m256 sz         = _mm256_mul_ps(dl, nz);
                sx                = _mm256_and_ps(sx, vmask);
                sy                = _mm256_and_ps(sy, vmask);
                sz                = _mm256_and_ps(sz, vmask);
                alignas(32) float sx_a[8], sy_a[8], sz_a[8], wi_a[8], wj_a[8], lambda_new_a[8];
                _mm256_store_ps(sx_a, sx);
                _mm256_store_ps(sy_a, sy);
                _mm256_store_ps(sz_a, sz);
                _mm256_store_ps(wi_a, wi);
                _mm256_store_ps(wj_a, wj);
                _mm256_store_ps(lambda_new_a, lambda_new);
                for (int lane = 0; lane < lane_count; ++lane) {
                    u32 e      = e_idx[lane];
                    u32 i      = (u32) idx_i[lane];
                    u32 j      = (u32) idx_j[lane];
                    float wi_s = wi_a[lane];
                    float wj_s = wj_a[lane];
                    if (wi_s + wj_s == 0.0f) continue;
                    float dix = wi_s * sx_a[lane];
                    float diy = wi_s * sy_a[lane];
                    float diz = wi_s * sz_a[lane];
                    float djx = -wj_s * sx_a[lane];
                    float djy = -wj_s * sy_a[lane];
                    float djz = -wj_s * sz_a[lane];
                    d.pos_x[i] += dix;
                    d.pos_y[i] += diy;
                    d.pos_z[i] += diz;
                    d.pos_x[j] += djx;
                    d.pos_y[j] += djy;
                    d.pos_z[j] += djz;
                    d.lambda_stretch[e] = lambda_new_a[lane];
                }
            }
        }
        inline void solve_stretch_batch_avx2_nc(const StaticState& s, DynamicState& d, const avec<u32>& batch) noexcept {
            if (batch.empty()) return;
            const __m256 v_eps = _mm256_set1_ps(k_epsilon);
            for (usize off = 0; off < batch.size(); off += 8) {
                alignas(32) u32 e_idx[8]{};
                int lane_count = (int) std::min<usize>(8, batch.size() - off);
                for (int lane = 0; lane < lane_count; ++lane) e_idx[lane] = batch[off + lane];
                alignas(32) int idx_i[8]{};
                alignas(32) int idx_j[8]{};
                for (int lane = 0; lane < lane_count; ++lane) {
                    u32 e       = e_idx[lane];
                    idx_i[lane] = (int) s.e_i[e];
                    idx_j[lane] = (int) s.e_j[e];
                }
                __m256i vi        = _mm256_load_si256(reinterpret_cast<const __m256i*>(idx_i));
                __m256i vj        = _mm256_load_si256(reinterpret_cast<const __m256i*>(idx_j));
                __m256 xi         = _mm256_i32gather_ps(d.pos_x.data(), vi, 4);
                __m256 yi         = _mm256_i32gather_ps(d.pos_y.data(), vi, 4);
                __m256 zi         = _mm256_i32gather_ps(d.pos_z.data(), vi, 4);
                __m256 xj         = _mm256_i32gather_ps(d.pos_x.data(), vj, 4);
                __m256 yj         = _mm256_i32gather_ps(d.pos_y.data(), vj, 4);
                __m256 zj         = _mm256_i32gather_ps(d.pos_z.data(), vj, 4);
                __m256 dx         = _mm256_sub_ps(xi, xj);
                __m256 dy         = _mm256_sub_ps(yi, yj);
                __m256 dz         = _mm256_sub_ps(zi, zj);
                __m256 len2       = _mm256_fmadd_ps(dz, dz, _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dx, dx)));
                __m256 mask_valid = _mm256_cmp_ps(len2, v_eps, _CMP_GT_OQ);
                __m256 wi         = _mm256_i32gather_ps(d.inv_mass.data(), vi, 4);
                __m256 wj         = _mm256_i32gather_ps(d.inv_mass.data(), vj, 4);
                __m256 wsum       = _mm256_add_ps(wi, wj);
                __m256 mask_mass  = _mm256_cmp_ps(wsum, _mm256_set1_ps(0.0f), _CMP_GT_OQ);
                __m256 vmask      = _mm256_and_ps(mask_valid, mask_mass);
                if (_mm256_testz_ps(vmask, vmask)) continue;
                __m256 len = _mm256_sqrt_ps(len2);
                alignas(32) int e_int[8]{};
                for (int lane = 0; lane < lane_count; ++lane) e_int[lane] = (int) e_idx[lane];
                __m256i ve      = _mm256_load_si256(reinterpret_cast<const __m256i*>(e_int));
                __m256 rest     = _mm256_i32gather_ps(s.rest_len.data(), ve, 4);
                __m256 C        = _mm256_sub_ps(len, rest);
                __m256 inv_wsum = _mm256_div_ps(_mm256_set1_ps(1.0f), wsum);
                __m256 dl       = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), C), inv_wsum);
                __m256 inv_len  = _mm256_div_ps(_mm256_set1_ps(1.0f), len);
                __m256 nx       = _mm256_mul_ps(dx, inv_len);
                __m256 ny       = _mm256_mul_ps(dy, inv_len);
                __m256 nz       = _mm256_mul_ps(dz, inv_len);
                __m256 sx       = _mm256_mul_ps(dl, nx);
                __m256 sy       = _mm256_mul_ps(dl, ny);
                __m256 sz       = _mm256_mul_ps(dl, nz);
                sx              = _mm256_and_ps(sx, vmask);
                sy              = _mm256_and_ps(sy, vmask);
                sz              = _mm256_and_ps(sz, vmask);
                alignas(32) float sx_a[8], sy_a[8], sz_a[8], wi_a[8], wj_a[8];
                _mm256_store_ps(sx_a, sx);
                _mm256_store_ps(sy_a, sy);
                _mm256_store_ps(sz_a, sz);
                _mm256_store_ps(wi_a, wi);
                _mm256_store_ps(wj_a, wj);
                for (int lane = 0; lane < lane_count; ++lane) {
                    u32 i      = (u32) idx_i[lane];
                    u32 j      = (u32) idx_j[lane];
                    float wi_s = wi_a[lane];
                    float wj_s = wj_a[lane];
                    if (wi_s + wj_s == 0.0f) continue;
                    float dix = wi_s * sx_a[lane];
                    float diy = wi_s * sy_a[lane];
                    float diz = wi_s * sz_a[lane];
                    float djx = -wj_s * sx_a[lane];
                    float djy = -wj_s * sy_a[lane];
                    float djz = -wj_s * sz_a[lane];
                    d.pos_x[i] += dix;
                    d.pos_y[i] += diy;
                    d.pos_z[i] += diz;
                    d.pos_x[j] += djx;
                    d.pos_y[j] += djy;
                    d.pos_z[j] += djz;
                }
            }
        }
        inline void step_avx2(Sim& sim, const StepParams& params) noexcept {
            auto& s             = sim.st;
            auto& d             = sim.dy;
            int sub             = std::max(1, s.solve.substeps);
            int iters           = std::max(1, s.solve.iterations);
            f32 full_dt         = params.dt > 0.0f ? params.dt : f32(1.0f / 60.0f);
            f32 dt              = full_dt / static_cast<f32>(sub);
            bool use_compliance = s.solve.compliance_stretch > 0.0f;
            for (int ss = 0; ss < sub; ++ss) {
                predict_positions(d, params, dt);
                if (use_compliance) std::fill(d.lambda_stretch.begin(), d.lambda_stretch.end(), 0.0f);
                for (int it = 0; it < iters; ++it) {
                    if (use_compliance) {
                        for (const auto& batch : s.batches) solve_stretch_batch_avx2(s, d, batch, s.solve.compliance_stretch, dt);
                    } else {
                        for (const auto& batch : s.batches) solve_stretch_batch_avx2_nc(s, d, batch);
                    }
                }
                integrate(d, dt, s.solve.damping);
            }
        }
#else
        // Fallback stub when AVX2 disabled at build time
        inline void step_avx2(Sim& sim, const StepParams& params) noexcept {
            step_native(sim, params);
        }
#endif

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
        if (h) {
            if (h->st.exec.backend == ExecPolicy::Backend::Avx2)
                detail::step_avx2(*h, params);
            else
                detail::step_native(*h, params);
        }
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
