// AVX2 XPBD implementation (decoupled)
#include "xpbd_avx2.h"
#include "cloth.h"

#ifdef HINAPE_HAVE_AVX2
#include <immintrin.h>
#endif

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

        class aligned_resource final : public std::pmr::memory_resource {
        public:
            explicit aligned_resource(std::size_t alignment = k_align, std::pmr::memory_resource* upstream = std::pmr::get_default_resource())
                : align_(normalize_alignment(alignment)), upstream_(upstream) {}

        private:
            std::size_t align_{};
            std::pmr::memory_resource* upstream_{};

            static std::size_t normalize_alignment(std::size_t a) {
                constexpr std::size_t base = alignof(void*);
                if (a < base) a = base;
                if ((a & (a - 1)) != 0) { std::size_t p = 1; while (p < a) p <<= 1U; a = p; }
                return a;
            }
            void* do_allocate(std::size_t bytes, std::size_t alignment) override {
                const std::size_t req = normalize_alignment(std::max(align_, alignment));
                if (req <= alignof(std::max_align_t)) {
                    return upstream_->allocate(bytes == 0 ? sizeof(std::max_align_t) : bytes, req);
                }
#if defined(_MSC_VER)
                void* p = _aligned_malloc(bytes == 0 ? req : bytes, req);
                if (!p) throw std::bad_alloc{};
                return p;
#else
                void* p = nullptr;
                const auto sz = static_cast<std::size_t>(bytes == 0 ? req : bytes);
                if (posix_memalign(&p, req, sz) != 0) throw std::bad_alloc{};
                return p;
#endif
            }
            void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
                if (!p) return;
                const std::size_t req = normalize_alignment(std::max(align_, alignment));
                if (req <= alignof(std::max_align_t)) { upstream_->deallocate(p, bytes == 0 ? sizeof(std::max_align_t) : bytes, req); return; }
#if defined(_MSC_VER)
                _aligned_free(p);
#else
                std::free(p);
#endif
            }
            [[nodiscard]] bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override { return this == &other; }
        };

        template <class T>
        using pvec = std::pmr::vector<T>;

        struct bitset_dyn {
            std::vector<std::uint64_t> w;
            void ensure(int bit) { usize need = static_cast<usize>(bit / 64 + 1); if (w.size() < need) w.resize(need, 0); }
            void set(int bit) { ensure(bit); w[static_cast<usize>(bit / 64)] |= (std::uint64_t(1) << (bit & 63)); }
            [[nodiscard]] bool test(int bit) const noexcept { usize idx = static_cast<usize>(bit / 64); if (idx >= w.size()) return false; return (w[idx] >> (bit & 63)) & 1ULL; }
        };

        struct BatchPack {
            pvec<int> i;
            pvec<int> j;
            pvec<float> rest;
            pvec<float> lambda; // only used when compliance > 0
            usize count{0};
            explicit BatchPack(std::pmr::memory_resource* mr) : i(mr), j(mr), rest(mr), lambda(mr) {}
        };

        struct StaticState {
            usize n_verts{0};
            usize n_edges{0};
            std::pmr::vector<BatchPack> batches;
            SolvePolicy solve{};
            ExecPolicy exec{};
            explicit StaticState(std::pmr::memory_resource* mr) : batches(mr) {}
        };
        struct DynamicState {
            pvec<f32> pos_x, pos_y, pos_z;
            pvec<f32> prev_x, prev_y, prev_z;
            pvec<f32> vel_x, vel_y, vel_z;
            pvec<f32> inv_mass;
            explicit DynamicState(std::pmr::memory_resource* mr)
                : pos_x(mr), pos_y(mr), pos_z(mr), prev_x(mr), prev_y(mr), prev_z(mr), vel_x(mr), vel_y(mr), vel_z(mr), inv_mass(mr) {}
        };

        class Avx2Sim final : public ISim {
        public:
            Avx2Sim() : mem64(k_align), arena(&mem64), st(&arena), dy(&arena) {}

            void init(const InitDesc& desc) {
                assert(desc.positions_xyz.size() % 3 == 0);
                const usize n = desc.positions_xyz.size() / 3;
                st.n_verts     = n;
                st.solve       = desc.solve;
                st.exec        = desc.exec;
                // vertices
                const usize n_pad = (n + 7u) & ~usize(7);
                dy.pos_x.resize(n_pad); dy.pos_y.resize(n_pad); dy.pos_z.resize(n_pad);
                dy.prev_x.resize(n_pad); dy.prev_y.resize(n_pad); dy.prev_z.resize(n_pad);
                dy.vel_x.resize(n_pad); dy.vel_y.resize(n_pad); dy.vel_z.resize(n_pad);
                dy.inv_mass.resize(n_pad);
                for (usize i = 0; i < n; ++i) {
                    dy.pos_x[i] = desc.positions_xyz[i * 3 + 0];
                    dy.pos_y[i] = desc.positions_xyz[i * 3 + 1];
                    dy.pos_z[i] = desc.positions_xyz[i * 3 + 2];
                    dy.prev_x[i] = dy.pos_x[i]; dy.prev_y[i] = dy.pos_y[i]; dy.prev_z[i] = dy.pos_z[i];
                    dy.vel_x[i] = dy.vel_y[i] = dy.vel_z[i] = 0.0f; dy.inv_mass[i] = 1.0f;
                }
                for (usize i = n; i < n_pad; ++i) { dy.pos_x[i] = dy.pos_y[i] = dy.pos_z[i] = 0.0f; dy.prev_x[i] = dy.prev_y[i] = dy.prev_z[i] = 0.0f; dy.vel_x[i] = dy.vel_y[i] = dy.vel_z[i] = 0.0f; dy.inv_mass[i] = 0.0f; }
                for (u32 idx : desc.fixed_indices) if (idx < n) dy.inv_mass[idx] = 0.0f;

                // edges -> batches -> packs (i, j, rest contiguous, padded to 8)
                std::vector<std::pair<u32, u32>> edges;
                build_edges_from_triangles(desc.triangles, edges);
                st.n_edges = edges.size();
                std::vector<std::vector<u32>> adj; build_adjacency(n, edges, adj);
                std::vector<std::vector<u32>> batches_idx; greedy_edge_coloring(n, edges, adj, batches_idx);
                st.batches.clear(); st.batches.reserve(batches_idx.size());
                for (const auto& b : batches_idx) {
                    BatchPack pack(&arena);
                    const usize m = b.size();
                    const usize padded = (m + 7u) & ~usize(7);
                    pack.count = m;
                    pack.i.resize(padded); pack.j.resize(padded); pack.rest.resize(padded);
                    if (st.solve.compliance_stretch > 0.0f) pack.lambda.resize(padded, 0.0f);
                    for (usize t = 0; t < m; ++t) {
                        u32 e = b[t];
                        u32 i = edges[e].first;
                        u32 j = edges[e].second;
                        pack.i[t] = static_cast<int>(i);
                        pack.j[t] = static_cast<int>(j);
                        float dx = desc.positions_xyz[i * 3 + 0] - desc.positions_xyz[j * 3 + 0];
                        float dy_ = desc.positions_xyz[i * 3 + 1] - desc.positions_xyz[j * 3 + 1];
                        float dz = desc.positions_xyz[i * 3 + 2] - desc.positions_xyz[j * 3 + 2];
                        pack.rest[t] = std::sqrt(dx * dx + dy_ * dy_ + dz * dz);
                    }
                    // pad lanes with i==j to disable without branching
                    int pin = 0; // any index; i==j ensures len2==0 -> skipped
                    for (usize t = m; t < padded; ++t) { pack.i[t] = pin; pack.j[t] = pin; pack.rest[t] = 0.0f; if (!pack.lambda.empty()) pack.lambda[t] = 0.0f; }
                    st.batches.emplace_back(std::move(pack));
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
                    if (use_compliance) zero_lambdas_();
                    for (int it = 0; it < iters; ++it) {
                        if (use_compliance)
                            for (auto& b : st.batches) solve_pack_compliance_(b, dt, st.solve.compliance_stretch);
                        else
                            for (auto& b : st.batches) solve_pack_(b);
                    }
                    integrate(dt, st.solve.damping);
                }
            }

            DynamicView map_dynamic() noexcept override {
                DynamicView v{}; v.pos_x = dy.pos_x.data(); v.pos_y = dy.pos_y.data(); v.pos_z = dy.pos_z.data(); v.vel_x = dy.vel_x.data(); v.vel_y = dy.vel_y.data(); v.vel_z = dy.vel_z.data(); v.count = st.n_verts; return v;
            }

        private:
            aligned_resource mem64;
            std::pmr::monotonic_buffer_resource arena;
            StaticState st;
            DynamicState dy;

            static void build_edges_from_triangles(std::span<const u32> tris, std::vector<std::pair<u32, u32>>& edges) {
                const usize m = tris.size() / 3; edges.clear(); edges.reserve(m * 3);
                for (usize t = 0; t < m; ++t) {
                    u32 a = tris[t * 3 + 0]; u32 b = tris[t * 3 + 1]; u32 c = tris[t * 3 + 2];
                    u32 e0a = a < b ? a : b; u32 e0b = a < b ? b : a;
                    u32 e1a = b < c ? b : c; u32 e1b = b < c ? c : b;
                    u32 e2a = c < a ? c : a; u32 e2b = c < a ? a : c;
                    edges.emplace_back(e0a, e0b); edges.emplace_back(e1a, e1b); edges.emplace_back(e2a, e2b);
                }
                std::sort(edges.begin(), edges.end()); edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
            }
            static void build_adjacency(usize n, const std::vector<std::pair<u32, u32>>& edges, std::vector<std::vector<u32>>& adj) {
                adj.assign(n, {});
                for (u32 e = 0; e < static_cast<u32>(edges.size()); ++e) { auto [i, j] = edges[e]; adj[i].push_back(e); adj[j].push_back(e); }
            }
            static void greedy_edge_coloring(usize n, const std::vector<std::pair<u32, u32>>& edges, const std::vector<std::vector<u32>>& adj, std::vector<std::vector<u32>>& batches) {
                const usize m = edges.size(); std::vector<u32> order(m); std::iota(order.begin(), order.end(), 0);
                std::vector<usize> deg(n); for (usize v = 0; v < n; ++v) deg[v] = adj[v].size();
                std::sort(order.begin(), order.end(), [&](u32 a, u32 b) {
                    auto [ai, aj] = edges[a]; auto [bi, bj] = edges[b];
                    usize da = std::max(deg[ai], deg[aj]); usize db = std::max(deg[bi], deg[bj]);
                    if (da != db) return da > db; return a < b;
                });
                std::vector<bitset_dyn> used(n); int maxc = -1; std::vector<int> color(m, -1);
                for (u32 idx : order) { auto [i, j] = edges[idx]; int c = 0; for (;;) { if (!used[i].test(c) && !used[j].test(c)) break; ++c; } used[i].set(c); used[j].set(c); color[idx] = c; if (c > maxc) maxc = c; }
                const usize nb = static_cast<usize>(maxc + 1); batches.assign(nb, {});
                for (u32 e = 0; e < static_cast<u32>(m); ++e) batches[static_cast<usize>(color[e])].push_back(e);
            }

            void zero_lambdas_() noexcept {
                if (st.solve.compliance_stretch <= 0.0f) return;
                for (auto& b : st.batches) std::fill(b.lambda.begin(), b.lambda.end(), 0.0f);
            }

            void predict_positions(const StepParams& p, f32 dt) noexcept {
                const f32 gx = p.gravity_x, gy = p.gravity_y, gz = p.gravity_z; const usize n = dy.inv_mass.size();
                for (usize i = 0; i < n; ++i) {
                    f32 im = dy.inv_mass[i]; dy.prev_x[i] = dy.pos_x[i]; dy.prev_y[i] = dy.pos_y[i]; dy.prev_z[i] = dy.pos_z[i];
                    if (im > 0.0f) { dy.vel_x[i] += gx * dt; dy.vel_y[i] += gy * dt; dy.vel_z[i] += gz * dt; dy.pos_x[i] += dy.vel_x[i] * dt; dy.pos_y[i] += dy.vel_y[i] * dt; dy.pos_z[i] += dy.vel_z[i] * dt; }
                    else { dy.vel_x[i] = dy.vel_y[i] = dy.vel_z[i] = 0.0f; }
                }
            }

#ifdef HINAPE_HAVE_AVX2
            void solve_pack_compliance_(BatchPack& b, f32 dt, f32 compliance) noexcept {
                const f32 alpha = compliance / (dt * dt);
                const __m256 v_alpha = _mm256_set1_ps(alpha);
                const __m256 v_eps   = _mm256_set1_ps(k_epsilon);
                const int padded     = static_cast<int>(b.i.size());
                for (int off = 0; off < padded; off += 8) {
                    const __m256i vi = _mm256_load_si256(reinterpret_cast<const __m256i*>(&b.i[off]));
                    const __m256i vj = _mm256_load_si256(reinterpret_cast<const __m256i*>(&b.j[off]));
                    __m256 xi = _mm256_i32gather_ps(dy.pos_x.data(), vi, 4);
                    __m256 yi = _mm256_i32gather_ps(dy.pos_y.data(), vi, 4);
                    __m256 zi = _mm256_i32gather_ps(dy.pos_z.data(), vi, 4);
                    __m256 xj = _mm256_i32gather_ps(dy.pos_x.data(), vj, 4);
                    __m256 yj = _mm256_i32gather_ps(dy.pos_y.data(), vj, 4);
                    __m256 zj = _mm256_i32gather_ps(dy.pos_z.data(), vj, 4);
                    __m256 dx = _mm256_sub_ps(xi, xj);
                    __m256 dy_ = _mm256_sub_ps(yi, yj);
                    __m256 dz = _mm256_sub_ps(zi, zj);
                    __m256 len2 = _mm256_fmadd_ps(dz, dz, _mm256_fmadd_ps(dy_, dy_, _mm256_mul_ps(dx, dx)));
                    __m256 mask_valid = _mm256_cmp_ps(len2, v_eps, _CMP_GT_OQ);
                    __m256 wi = _mm256_i32gather_ps(dy.inv_mass.data(), vi, 4);
                    __m256 wj = _mm256_i32gather_ps(dy.inv_mass.data(), vj, 4);
                    __m256 wsum = _mm256_add_ps(wi, wj);
                    __m256 mask_mass = _mm256_cmp_ps(wsum, _mm256_set1_ps(0.0f), _CMP_GT_OQ);
                    __m256 vmask = _mm256_and_ps(mask_valid, mask_mass);
                    if (_mm256_testz_ps(vmask, vmask)) continue;
                    __m256 len = _mm256_sqrt_ps(len2);
                    __m256 rest = _mm256_load_ps(&b.rest[off]);
                    __m256 C    = _mm256_sub_ps(len, rest);
                    __m256 lambda_old = _mm256_load_ps(&b.lambda[off]);
                    __m256 denom      = _mm256_add_ps(wsum, v_alpha);
                    __m256 tmp        = _mm256_fmadd_ps(v_alpha, lambda_old, C);
                    __m256 dl         = _mm256_div_ps(_mm256_sub_ps(_mm256_setzero_ps(), tmp), denom);
                    __m256 lambda_new = _mm256_add_ps(lambda_old, dl);
                    __m256 inv_len    = _mm256_div_ps(_mm256_set1_ps(1.0f), len);
                    __m256 nx         = _mm256_mul_ps(dx, inv_len);
                    __m256 ny         = _mm256_mul_ps(dy_, inv_len);
                    __m256 nz         = _mm256_mul_ps(dz, inv_len);
                    __m256 sx         = _mm256_mul_ps(dl, nx);
                    __m256 sy         = _mm256_mul_ps(dl, ny);
                    __m256 sz         = _mm256_mul_ps(dl, nz);
                    sx = _mm256_and_ps(sx, vmask); sy = _mm256_and_ps(sy, vmask); sz = _mm256_and_ps(sz, vmask);
                    alignas(32) float sx_a[8], sy_a[8], sz_a[8], wi_a[8], wj_a[8], lambda_new_a[8];
                    _mm256_store_ps(sx_a, sx); _mm256_store_ps(sy_a, sy); _mm256_store_ps(sz_a, sz);
                    _mm256_store_ps(wi_a, wi); _mm256_store_ps(wj_a, wj); _mm256_store_ps(lambda_new_a, lambda_new);
                    for (int lane = 0; lane < 8; ++lane) {
                        int i = b.i[off + lane]; int j = b.j[off + lane];
                        float wi_s = wi_a[lane]; float wj_s = wj_a[lane];
                        if (wi_s + wj_s == 0.0f) continue;
                        float dix = wi_s * sx_a[lane]; float diy = wi_s * sy_a[lane]; float diz = wi_s * sz_a[lane];
                        float djx = -wj_s * sx_a[lane]; float djy = -wj_s * sy_a[lane]; float djz = -wj_s * sz_a[lane];
                        dy.pos_x[(usize)i] += dix; dy.pos_y[(usize)i] += diy; dy.pos_z[(usize)i] += diz;
                        dy.pos_x[(usize)j] += djx; dy.pos_y[(usize)j] += djy; dy.pos_z[(usize)j] += djz;
                        b.lambda[off + lane] = lambda_new_a[lane];
                    }
                }
            }
            void solve_pack_(BatchPack& b) noexcept {
                const __m256 v_eps = _mm256_set1_ps(k_epsilon);
                const int padded   = static_cast<int>(b.i.size());
                for (int off = 0; off < padded; off += 8) {
                    const __m256i vi = _mm256_load_si256(reinterpret_cast<const __m256i*>(&b.i[off]));
                    const __m256i vj = _mm256_load_si256(reinterpret_cast<const __m256i*>(&b.j[off]));
                    __m256 xi = _mm256_i32gather_ps(dy.pos_x.data(), vi, 4);
                    __m256 yi = _mm256_i32gather_ps(dy.pos_y.data(), vi, 4);
                    __m256 zi = _mm256_i32gather_ps(dy.pos_z.data(), vi, 4);
                    __m256 xj = _mm256_i32gather_ps(dy.pos_x.data(), vj, 4);
                    __m256 yj = _mm256_i32gather_ps(dy.pos_y.data(), vj, 4);
                    __m256 zj = _mm256_i32gather_ps(dy.pos_z.data(), vj, 4);
                    __m256 dx = _mm256_sub_ps(xi, xj);
                    __m256 dy_ = _mm256_sub_ps(yi, yj);
                    __m256 dz = _mm256_sub_ps(zi, zj);
                    __m256 len2 = _mm256_fmadd_ps(dz, dz, _mm256_fmadd_ps(dy_, dy_, _mm256_mul_ps(dx, dx)));
                    __m256 mask_valid = _mm256_cmp_ps(len2, v_eps, _CMP_GT_OQ);
                    __m256 wi = _mm256_i32gather_ps(dy.inv_mass.data(), vi, 4);
                    __m256 wj = _mm256_i32gather_ps(dy.inv_mass.data(), vj, 4);
                    __m256 wsum = _mm256_add_ps(wi, wj);
                    __m256 mask_mass = _mm256_cmp_ps(wsum, _mm256_set1_ps(0.0f), _CMP_GT_OQ);
                    __m256 vmask = _mm256_and_ps(mask_valid, mask_mass);
                    if (_mm256_testz_ps(vmask, vmask)) continue;
                    __m256 len = _mm256_sqrt_ps(len2);
                    __m256 rest = _mm256_load_ps(&b.rest[off]);
                    __m256 C = _mm256_sub_ps(len, rest);
                    __m256 inv_wsum = _mm256_div_ps(_mm256_set1_ps(1.0f), wsum);
                    __m256 dl = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), C), inv_wsum);
                    __m256 inv_len = _mm256_div_ps(_mm256_set1_ps(1.0f), len);
                    __m256 nx = _mm256_mul_ps(dx, inv_len);
                    __m256 ny = _mm256_mul_ps(dy_, inv_len);
                    __m256 nz = _mm256_mul_ps(dz, inv_len);
                    __m256 sx = _mm256_mul_ps(dl, nx);
                    __m256 sy = _mm256_mul_ps(dl, ny);
                    __m256 sz = _mm256_mul_ps(dl, nz);
                    sx = _mm256_and_ps(sx, vmask); sy = _mm256_and_ps(sy, vmask); sz = _mm256_and_ps(sz, vmask);
                    alignas(32) float sx_a[8], sy_a[8], sz_a[8], wi_a[8], wj_a[8];
                    _mm256_store_ps(sx_a, sx); _mm256_store_ps(sy_a, sy); _mm256_store_ps(sz_a, sz);
                    _mm256_store_ps(wi_a, wi); _mm256_store_ps(wj_a, wj);
                    for (int lane = 0; lane < 8; ++lane) {
                        int i = b.i[off + lane]; int j = b.j[off + lane];
                        float wi_s = wi_a[lane]; float wj_s = wj_a[lane]; if (wi_s + wj_s == 0.0f) continue;
                        float dix = wi_s * sx_a[lane]; float diy = wi_s * sy_a[lane]; float diz = wi_s * sz_a[lane];
                        float djx = -wj_s * sx_a[lane]; float djy = -wj_s * sy_a[lane]; float djz = -wj_s * sz_a[lane];
                        dy.pos_x[(usize)i] += dix; dy.pos_y[(usize)i] += diy; dy.pos_z[(usize)i] += diz;
                        dy.pos_x[(usize)j] += djx; dy.pos_y[(usize)j] += djy; dy.pos_z[(usize)j] += djz;
                    }
                }
            }
#else
            void solve_pack_compliance_(BatchPack&, f32, f32) noexcept {}
            void solve_pack_(BatchPack&) noexcept {}
#endif

            void integrate(f32 dt, f32 damping) noexcept {
                const usize n = dy.inv_mass.size(); const f32 k = std::clamp(1.0f - damping, 0.0f, 1.0f);
                for (usize i = 0; i < n; ++i) {
                    if (dy.inv_mass[i] > 0.0f) {
                        f32 vx = (dy.pos_x[i] - dy.prev_x[i]) / dt; f32 vy = (dy.pos_y[i] - dy.prev_y[i]) / dt; f32 vz = (dy.pos_z[i] - dy.prev_z[i]) / dt;
                        dy.vel_x[i] = vx * k; dy.vel_y[i] = vy * k; dy.vel_z[i] = vz * k;
                    } else {
                        dy.vel_x[i] = dy.vel_y[i] = dy.vel_z[i] = 0.0f; dy.pos_x[i] = dy.prev_x[i]; dy.pos_y[i] = dy.prev_y[i]; dy.pos_z[i] = dy.prev_z[i];
                    }
                }
            }
        };

#if defined(HINAPE_HAVE_AVX2)
        ISim* make_avx2(const InitDesc& desc) {
            auto* s = new Avx2Sim(); s->init(desc); return s;
        }
#endif

    } // namespace detail
} // namespace HinaPE


