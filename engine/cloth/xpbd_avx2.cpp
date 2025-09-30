#include "cloth.h"
#include "cloth/core/arena.h"
#include "cloth/core/topology.h"
#include "cloth/model/cloth_data.h"

#if defined(HINAPE_HAVE_AVX2)
#include <immintrin.h>
#endif

#include <algorithm>
#include <cmath>
#include <memory_resource>
#include <numeric>
#include <vector>

namespace HinaPE {
namespace detail {

#if defined(HINAPE_HAVE_AVX2)

using std::size_t;

struct Avx2Static {
    usize n_verts{0};
    std::pmr::vector<std::pmr::vector<u32>> batches;
    SolvePolicy solve{};
    ExecPolicy exec{};

    explicit Avx2Static(std::pmr::memory_resource* mr) : batches(mr) {}
};

struct Avx2Dynamic {
    model::ClothData data;
    std::pmr::vector<float> prev_x, prev_y, prev_z, lambda;

    explicit Avx2Dynamic(std::pmr::memory_resource* mr)
        : data(mr), prev_x(mr), prev_y(mr), prev_z(mr), lambda(mr) {}
};

class Avx2Sim final : public ISim {
public:
    Avx2Sim() : mem64(64), arena(&mem64), st(&arena), dy(&arena) {}

    void init(const InitDesc& desc) {
        const usize n = desc.positions_xyz.size() / 3;

        st.n_verts = n;
        st.solve = desc.solve;
        st.exec = desc.exec;

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
            dy.prev_x[i] = dy.data.px[i];
            dy.prev_y[i] = dy.data.py[i];
            dy.prev_z[i] = dy.data.pz[i];
            dy.data.vx[i] = dy.data.vy[i] = dy.data.vz[i] = 0.0f;
            dy.data.inv_mass[i] = 1.0f;
        }

        for (u32 f : desc.fixed_indices) {
            if (f < n) {
                dy.data.inv_mass[f] = 0.0f;
            }
        }

        for (usize k = 0; k < edges.size(); ++k) {
            dy.data.e_i[k] = edges[k].first;
            dy.data.e_j[k] = edges[k].second;
        }

        for (usize k = 0; k < edges.size(); ++k) {
            u32 i = dy.data.e_i[k], j = dy.data.e_j[k];
            float dx = dy.data.px[i] - dy.data.px[j];
            float dy_ = dy.data.py[i] - dy.data.py[j];
            float dz = dy.data.pz[i] - dy.data.pz[j];
            dy.data.rest_len[k] = std::sqrt(dx * dx + dy_ * dy_ + dz * dz);
        }

        if (st.solve.compliance_stretch > 0.0f) {
            dy.lambda.assign(dy.data.e_i.size(), 0.0f);
        }
    }

    void step(const StepParams& params) noexcept override {
        const int sub = std::max(1, st.solve.substeps);
        const int iters = std::max(1, st.solve.iterations);

        const f32 full_dt = params.dt > 0.0f ? params.dt : f32(1.0f / 60.0f);
        const f32 dt = full_dt / static_cast<f32>(sub);
        const bool useC = st.solve.compliance_stretch > 0.0f;

        for (int s = 0; s < sub; ++s) {
            predict(params, dt);
            if (useC) {
                std::fill(dy.lambda.begin(), dy.lambda.end(), 0.0f);
            }
            for (int it = 0; it < iters; ++it) {
                if (useC) {
                    for (auto& b : st.batches) {
                        solve_batch_c(b, dt, st.solve.compliance_stretch);
                    }
                } else {
                    for (auto& b : st.batches) {
                        solve_batch(b);
                    }
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
    void predict(const StepParams& p, f32 dt) noexcept {
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

    void solve_batch_c(const std::pmr::vector<u32>& batch, f32 dt, f32 compliance) noexcept {
        if (batch.empty()) {
            return;
        }

        const f32 alpha = compliance / (dt * dt);
        const __m256 v_alpha = _mm256_set1_ps(alpha);
        const __m256 v_eps = _mm256_set1_ps(1e-8f);

        for (usize off = 0; off < batch.size(); off += 8) {
            alignas(32) int ii[8]{};
            alignas(32) int jj[8]{};
            alignas(32) int ee[8]{};
            const int lanes = (int)std::min<usize>(8, batch.size() - off);
            for (int l = 0; l < lanes; ++l) {
                u32 e = batch[off + l];
                ee[l] = (int)e;
                ii[l] = (int)dy.data.e_i[e];
                jj[l] = (int)dy.data.e_j[e];
            }

            __m256i vi = _mm256_load_si256((const __m256i*)ii);
            __m256i vj = _mm256_load_si256((const __m256i*)jj);
            __m256i ve = _mm256_load_si256((const __m256i*)ee);

            __m256 xi = _mm256_i32gather_ps(dy.data.px.data(), vi, 4);
            __m256 yi = _mm256_i32gather_ps(dy.data.py.data(), vi, 4);
            __m256 zi = _mm256_i32gather_ps(dy.data.pz.data(), vi, 4);

            __m256 xj = _mm256_i32gather_ps(dy.data.px.data(), vj, 4);
            __m256 yj = _mm256_i32gather_ps(dy.data.py.data(), vj, 4);
            __m256 zj = _mm256_i32gather_ps(dy.data.pz.data(), vj, 4);

            __m256 dx = _mm256_sub_ps(xi, xj);
            __m256 dy_ = _mm256_sub_ps(yi, yj);
            __m256 dz = _mm256_sub_ps(zi, zj);

            __m256 len2 = _mm256_fmadd_ps(dz, dz, _mm256_fmadd_ps(dy_, dy_, _mm256_mul_ps(dx, dx)));
            __m256 mask_valid = _mm256_cmp_ps(len2, v_eps, _CMP_GT_OQ);

            __m256 wi = _mm256_i32gather_ps(dy.data.inv_mass.data(), vi, 4);
            __m256 wj = _mm256_i32gather_ps(dy.data.inv_mass.data(), vj, 4);
            __m256 wsum = _mm256_add_ps(wi, wj);
            __m256 mask_mass = _mm256_cmp_ps(wsum, _mm256_set1_ps(0.0f), _CMP_GT_OQ);
            __m256 vmask = _mm256_and_ps(mask_valid, mask_mass);

            if (_mm256_testz_ps(vmask, vmask)) {
                continue;
            }

            __m256 len = _mm256_sqrt_ps(len2);
            __m256 rest = _mm256_i32gather_ps(dy.data.rest_len.data(), ve, 4);
            __m256 C = _mm256_sub_ps(len, rest);
            __m256 lambda_old = _mm256_i32gather_ps(dy.lambda.data(), ve, 4);
            __m256 denom = _mm256_add_ps(wsum, v_alpha);
            __m256 tmp = _mm256_fmadd_ps(v_alpha, lambda_old, C);
            __m256 dl = _mm256_div_ps(_mm256_sub_ps(_mm256_setzero_ps(), tmp), denom);
            __m256 lambda_new = _mm256_add_ps(lambda_old, dl);

            __m256 inv_len = _mm256_div_ps(_mm256_set1_ps(1.0f), len);
            __m256 nx = _mm256_mul_ps(dx, inv_len);
            __m256 ny = _mm256_mul_ps(dy_, inv_len);
            __m256 nz = _mm256_mul_ps(dz, inv_len);

            __m256 sx = _mm256_mul_ps(dl, nx);
            __m256 sy = _mm256_mul_ps(dl, ny);
            __m256 sz = _mm256_mul_ps(dl, nz);

            sx = _mm256_and_ps(sx, vmask);
            sy = _mm256_and_ps(sy, vmask);
            sz = _mm256_and_ps(sz, vmask);

            alignas(32) float sx_a[8], sy_a[8], sz_a[8], wi_a[8], wj_a[8], lnew_a[8];
            _mm256_store_ps(sx_a, sx);
            _mm256_store_ps(sy_a, sy);
            _mm256_store_ps(sz_a, sz);
            _mm256_store_ps(wi_a, wi);
            _mm256_store_ps(wj_a, wj);
            _mm256_store_ps(lnew_a, lambda_new);

            for (int l = 0; l < lanes; ++l) {
                u32 e = (u32)ee[l];
                u32 i = (u32)ii[l], j = (u32)jj[l];
                float wi_s = wi_a[l], wj_s = wj_a[l];
                if (wi_s + wj_s == 0.0f) {
                    continue;
                }
                float dix = wi_s * sx_a[l], diy = wi_s * sy_a[l], diz = wi_s * sz_a[l];
                float djx = -wj_s * sx_a[l], djy = -wj_s * sy_a[l], djz = -wj_s * sz_a[l];
                dy.data.px[i] += dix;
                dy.data.py[i] += diy;
                dy.data.pz[i] += diz;
                dy.data.px[j] += djx;
                dy.data.py[j] += djy;
                dy.data.pz[j] += djz;
                dy.lambda[e] = lnew_a[l];
            }
        }
    }

    void solve_batch(const std::pmr::vector<u32>& batch) noexcept {
        if (batch.empty()) {
            return;
        }

        const __m256 v_eps = _mm256_set1_ps(1e-8f);
        for (usize off = 0; off < batch.size(); off += 8) {
            alignas(32) int ii[8]{};
            alignas(32) int jj[8]{};
            alignas(32) int ee[8]{};
            const int lanes = (int)std::min<usize>(8, batch.size() - off);
            for (int l = 0; l < lanes; ++l) {
                u32 e = batch[off + l];
                ee[l] = (int)e;
                ii[l] = (int)dy.data.e_i[e];
                jj[l] = (int)dy.data.e_j[e];
            }

            __m256i vi = _mm256_load_si256((const __m256i*)ii);
            __m256i vj = _mm256_load_si256((const __m256i*)jj);
            __m256i ve = _mm256_load_si256((const __m256i*)ee);

            __m256 xi = _mm256_i32gather_ps(dy.data.px.data(), vi, 4);
            __m256 yi = _mm256_i32gather_ps(dy.data.py.data(), vi, 4);
            __m256 zi = _mm256_i32gather_ps(dy.data.pz.data(), vi, 4);

            __m256 xj = _mm256_i32gather_ps(dy.data.px.data(), vj, 4);
            __m256 yj = _mm256_i32gather_ps(dy.data.py.data(), vj, 4);
            __m256 zj = _mm256_i32gather_ps(dy.data.pz.data(), vj, 4);

            __m256 dx = _mm256_sub_ps(xi, xj);
            __m256 dy_ = _mm256_sub_ps(yi, yj);
            __m256 dz = _mm256_sub_ps(zi, zj);

            __m256 len2 = _mm256_fmadd_ps(dz, dz, _mm256_fmadd_ps(dy_, dy_, _mm256_mul_ps(dx, dx)));
            __m256 mask_valid = _mm256_cmp_ps(len2, v_eps, _CMP_GT_OQ);

            __m256 wi = _mm256_i32gather_ps(dy.data.inv_mass.data(), vi, 4);
            __m256 wj = _mm256_i32gather_ps(dy.data.inv_mass.data(), vj, 4);
            __m256 wsum = _mm256_add_ps(wi, wj);
            __m256 mask_mass = _mm256_cmp_ps(wsum, _mm256_set1_ps(0.0f), _CMP_GT_OQ);
            __m256 vmask = _mm256_and_ps(mask_valid, mask_mass);

            if (_mm256_testz_ps(vmask, vmask)) {
                continue;
            }

            __m256 len = _mm256_sqrt_ps(len2);
            __m256 inv_len = _mm256_div_ps(_mm256_set1_ps(1.0f), len);
            __m256 rest = _mm256_i32gather_ps(dy.data.rest_len.data(), ve, 4);
            __m256 C = _mm256_sub_ps(len, rest);
            __m256 dl = _mm256_div_ps(_mm256_sub_ps(_mm256_setzero_ps(), C), wsum);

            __m256 nx = _mm256_mul_ps(dx, inv_len);
            __m256 ny = _mm256_mul_ps(dy_, inv_len);
            __m256 nz = _mm256_mul_ps(dz, inv_len);

            __m256 sx = _mm256_mul_ps(dl, nx);
            __m256 sy = _mm256_mul_ps(dl, ny);
            __m256 sz = _mm256_mul_ps(dl, nz);

            sx = _mm256_and_ps(sx, vmask);
            sy = _mm256_and_ps(sy, vmask);
            sz = _mm256_and_ps(sz, vmask);

            alignas(32) float sx_a[8], sy_a[8], sz_a[8], wi_a[8], wj_a[8];
            _mm256_store_ps(sx_a, sx);
            _mm256_store_ps(sy_a, sy);
            _mm256_store_ps(sz_a, sz);
            _mm256_store_ps(wi_a, wi);
            _mm256_store_ps(wj_a, wj);

            for (int l = 0; l < lanes; ++l) {
                u32 i = (u32)ii[l], j = (u32)jj[l];
                float wi_s = wi_a[l], wj_s = wj_a[l];
                if (wi_s + wj_s == 0.0f) {
                    continue;
                }
                float dix = wi_s * sx_a[l], diy = wi_s * sy_a[l], diz = wi_s * sz_a[l];
                float djx = -wj_s * sx_a[l], djy = -wj_s * sy_a[l], djz = -wj_s * sz_a[l];
                dy.data.px[i] += dix;
                dy.data.py[i] += diy;
                dy.data.pz[i] += diz;
                dy.data.px[j] += djx;
                dy.data.py[j] += djy;
                dy.data.pz[j] += djz;
            }
        }
    }

    void integrate(f32 dt, f32 damping) noexcept {
        const usize n = dy.data.inv_mass.size();
        const f32 k = std::clamp(1.0f - damping, 0.0f, 1.0f);
        for (usize i = 0; i < n; ++i) {
            if (dy.data.inv_mass[i] > 0.0f) {
                f32 vx = (dy.data.px[i] - dy.prev_x[i]) / dt;
                f32 vy = (dy.data.py[i] - dy.prev_y[i]) / dt;
                f32 vz = (dy.data.pz[i] - dy.prev_z[i]) / dt;
                dy.data.vx[i] = vx * k;
                dy.data.vy[i] = vy * k;
                dy.data.vz[i] = vz * k;
            } else {
                dy.data.vx[i] = dy.data.vy[i] = dy.data.vz[i] = 0.0f;
                dy.data.px[i] = dy.prev_x[i];
                dy.data.py[i] = dy.prev_y[i];
                dy.data.pz[i] = dy.prev_z[i];
            }
        }
    }

    core::aligned_resource mem64;
    std::pmr::monotonic_buffer_resource arena;
    Avx2Static st;
    Avx2Dynamic dy;
};

ISim* make_avx2(const InitDesc& desc) {
    auto* s = new Avx2Sim();
    s->init(desc);
    return s;
}

#else

ISim* make_avx2(const InitDesc& desc) {
    return make_native(desc);
}

#endif

} // namespace detail
} // namespace HinaPE

