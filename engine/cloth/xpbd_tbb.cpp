#include "cloth.h"
#include "cloth/core/arena.h"
#include "cloth/core/topology.h"
#include "cloth/model/cloth_data.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <memory_resource>
#include <numeric>
#include <vector>

#if defined(HINAPE_HAVE_TBB)
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/info.h>
#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>
#endif

namespace HinaPE {
namespace detail {

using std::size_t;
constexpr f32 k_epsilon = 1e-8f;

#if defined(HINAPE_HAVE_TBB)

struct TbbStaticState {
    usize n_verts{0};
    std::pmr::vector<std::pmr::vector<u32>> batches;
    SolvePolicy solve{};
    ExecPolicy exec{};
    explicit TbbStaticState(std::pmr::memory_resource* mr) : batches(mr) {}
};

struct TbbDynamicState {
    model::ClothData data;
    std::pmr::vector<float> prev_x, prev_y, prev_z, lambda;
    explicit TbbDynamicState(std::pmr::memory_resource* mr)
        : data(mr), prev_x(mr), prev_y(mr), prev_z(mr), lambda(mr) {}
};

class TbbSim final : public ISim {
public:
    TbbSim() : mem64(64), arena(&mem64), st(&arena), dy(&arena) {}

    void init(const InitDesc& desc) {
        assert(desc.positions_xyz.size() % 3 == 0);
        const usize n = desc.positions_xyz.size() / 3;
        st.n_verts     = n;
        st.solve       = desc.solve;
        st.exec        = desc.exec;

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

        // Resource scope for threads control
        if (st.exec.threads > 0) {
            control = std::make_unique<tbb::global_control>(
                tbb::global_control::max_allowed_parallelism,
                static_cast<size_t>(st.exec.threads));
        } else {
            control.reset();
        }

        dy.data.resize_particles(n, true);
        dy.prev_x.resize(dy.data.px.size());
        dy.prev_y.resize(dy.data.py.size());
        dy.prev_z.resize(dy.data.pz.size());
        dy.data.resize_edges(edges.size());

        // Initialize particles (parallel over padded range)
        const usize n_pad = dy.data.px.size();
        {
            tbb::static_partitioner part;
            tbb::parallel_for(tbb::blocked_range<usize>(0, n_pad), [&](const tbb::blocked_range<usize>& r) {
                for (usize i = r.begin(); i != r.end(); ++i) {
                    float px = 0.0f, py = 0.0f, pz = 0.0f;
                    if (i < n) {
                        px = desc.positions_xyz[i * 3 + 0];
                        py = desc.positions_xyz[i * 3 + 1];
                        pz = desc.positions_xyz[i * 3 + 2];
                    }
                    dy.data.px[i] = px;
                    dy.data.py[i] = py;
                    dy.data.pz[i] = pz;
                    dy.prev_x[i]  = px;
                    dy.prev_y[i]  = py;
                    dy.prev_z[i]  = pz;
                    dy.data.vx[i] = 0.0f;
                    dy.data.vy[i] = 0.0f;
                    dy.data.vz[i] = 0.0f;
                    dy.data.inv_mass[i] = (i < n) ? 1.0f : 0.0f;
                }
            }, part);
        }

        for (u32 f : desc.fixed_indices) {
            if (f < n) dy.data.inv_mass[f] = 0.0f;
        }

        // Copy edges and compute rest lengths in parallel
        const usize m = edges.size();
        {
            tbb::static_partitioner part;
            tbb::parallel_for(tbb::blocked_range<usize>(0, m), [&](const tbb::blocked_range<usize>& r) {
                for (usize k = r.begin(); k != r.end(); ++k) {
                    dy.data.e_i[k] = edges[k].first;
                    dy.data.e_j[k] = edges[k].second;
                    u32 i = dy.data.e_i[k];
                    u32 j = dy.data.e_j[k];
                    float dx = dy.data.px[i] - dy.data.px[j];
                    float dy_ = dy.data.py[i] - dy.data.py[j];
                    float dz = dy.data.pz[i] - dy.data.pz[j];
                    dy.data.rest_len[k] = std::sqrt(dx * dx + dy_ * dy_ + dz * dz);
                }
            }, part);
        }

        if (st.solve.compliance_stretch > 0.0f) {
            dy.lambda.assign(m, 0.0f);
        }
    }

    void step(const StepParams& params) noexcept override {
        const int sub   = std::max(1, st.solve.substeps);
        const int iters = std::max(1, st.solve.iterations);
        const f32 full_dt = params.dt > 0.0f ? params.dt : f32(1.0f / 60.0f);
        const f32 dt = full_dt / static_cast<f32>(sub);
        const bool use_compliance = st.solve.compliance_stretch > 0.0f;

        for (int s = 0; s < sub; ++s) {
            predict_positions(params, dt);
            if (use_compliance) {
                std::fill(dy.lambda.begin(), dy.lambda.end(), 0.0f);
            }
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
    TbbStaticState st;
    TbbDynamicState dy;
    std::unique_ptr<tbb::global_control> control;

    void predict_positions(const StepParams& p, f32 dt) noexcept {
        const f32 gx = p.gravity_x, gy = p.gravity_y, gz = p.gravity_z;
        const usize n = dy.data.inv_mass.size();
        tbb::static_partitioner part;
        tbb::parallel_for(tbb::blocked_range<usize>(0, n), [&](const tbb::blocked_range<usize>& r) {
            auto* px = std::assume_aligned<64>(dy.data.px.data());
            auto* py = std::assume_aligned<64>(dy.data.py.data());
            auto* pz = std::assume_aligned<64>(dy.data.pz.data());
            auto* vx = std::assume_aligned<64>(dy.data.vx.data());
            auto* vy = std::assume_aligned<64>(dy.data.vy.data());
            auto* vz = std::assume_aligned<64>(dy.data.vz.data());
            auto* im = std::assume_aligned<64>(dy.data.inv_mass.data());
            auto* px_prev = std::assume_aligned<64>(dy.prev_x.data());
            auto* py_prev = std::assume_aligned<64>(dy.prev_y.data());
            auto* pz_prev = std::assume_aligned<64>(dy.prev_z.data());
            for (usize i = r.begin(); i != r.end(); ++i) {
                const f32 imi = im[i];
                px_prev[i] = px[i];
                py_prev[i] = py[i];
                pz_prev[i] = pz[i];
                if (imi > 0.0f) {
                    vx[i] += gx * dt;
                    vy[i] += gy * dt;
                    vz[i] += gz * dt;
                    px[i] += vx[i] * dt;
                    py[i] += vy[i] * dt;
                    pz[i] += vz[i] * dt;
                } else {
                    vx[i] = vy[i] = vz[i] = 0.0f;
                }
            }
        }, part);
    }

    void solve_stretch_batch(const std::pmr::vector<u32>& batch, f32 compliance, f32 dt) noexcept {
        if (batch.empty()) return;
        const f32 alpha = compliance / (dt * dt);
        tbb::static_partitioner part;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, batch.size()), [&](const tbb::blocked_range<size_t>& r) {
            auto* px = std::assume_aligned<64>(dy.data.px.data());
            auto* py = std::assume_aligned<64>(dy.data.py.data());
            auto* pz = std::assume_aligned<64>(dy.data.pz.data());
            auto* im = std::assume_aligned<64>(dy.data.inv_mass.data());
            auto* rest = std::assume_aligned<64>(dy.data.rest_len.data());
            auto* lambda = std::assume_aligned<64>(dy.lambda.data());
            for (size_t idx = r.begin(); idx != r.end(); ++idx) {
                u32 e = batch[idx];
                u32 i = dy.data.e_i[e], j = dy.data.e_j[e];
                f32 wi = im[i], wj = im[j];
                if (wi + wj == 0.0f) continue;
                f32 dx = px[i] - px[j];
                f32 dy_ = py[i] - py[j];
                f32 dz = pz[i] - pz[j];
                f32 len2 = dx * dx + dy_ * dy_ + dz * dz;
                if (len2 < k_epsilon) continue;
                f32 len = std::sqrt(len2);
                f32 C = len - rest[e];
                f32 inv_len = 1.0f / len;
                f32 dl = -(C + alpha * lambda[e]) / (wi + wj + alpha);
                lambda[e] += dl;
                f32 sx = dl * dx * inv_len, sy = dl * dy_ * inv_len, sz = dl * dz * inv_len;
                px[i] += wi * sx; py[i] += wi * sy; pz[i] += wi * sz;
                px[j] -= wj * sx; py[j] -= wj * sy; pz[j] -= wj * sz;
            }
        }, part);
    }

    void solve_stretch_batch_nc(const std::pmr::vector<u32>& batch) noexcept {
        if (batch.empty()) return;
        tbb::static_partitioner part;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, batch.size()), [&](const tbb::blocked_range<size_t>& r) {
            auto* px = std::assume_aligned<64>(dy.data.px.data());
            auto* py = std::assume_aligned<64>(dy.data.py.data());
            auto* pz = std::assume_aligned<64>(dy.data.pz.data());
            auto* im = std::assume_aligned<64>(dy.data.inv_mass.data());
            auto* rest = std::assume_aligned<64>(dy.data.rest_len.data());
            for (size_t idx = r.begin(); idx != r.end(); ++idx) {
                u32 e = batch[idx];
                u32 i = dy.data.e_i[e], j = dy.data.e_j[e];
                f32 wi = im[i], wj = im[j];
                if (wi + wj == 0.0f) continue;
                f32 dx = px[i] - px[j];
                f32 dy_ = py[i] - py[j];
                f32 dz = pz[i] - pz[j];
                f32 len2 = dx * dx + dy_ * dy_ + dz * dz;
                if (len2 < k_epsilon) continue;
                f32 len = std::sqrt(len2);
                f32 inv_len = 1.0f / len;
                f32 C = len - rest[e];
                f32 dl = -C / (wi + wj);
                f32 sx = dl * dx * inv_len, sy = dl * dy_ * inv_len, sz = dl * dz * inv_len;
                px[i] += wi * sx; py[i] += wi * sy; pz[i] += wi * sz;
                px[j] -= wj * sx; py[j] -= wj * sy; pz[j] -= wj * sz;
            }
        }, part);
    }

    void integrate(f32 dt, f32 damping) noexcept {
        const usize n = dy.data.inv_mass.size();
        const f32 k = std::clamp(1.0f - damping, 0.0f, 1.0f);
        tbb::static_partitioner part;
        tbb::parallel_for(tbb::blocked_range<usize>(0, n), [&](const tbb::blocked_range<usize>& r) {
            auto* px = std::assume_aligned<64>(dy.data.px.data());
            auto* py = std::assume_aligned<64>(dy.data.py.data());
            auto* pz = std::assume_aligned<64>(dy.data.pz.data());
            auto* vx = std::assume_aligned<64>(dy.data.vx.data());
            auto* vy = std::assume_aligned<64>(dy.data.vy.data());
            auto* vz = std::assume_aligned<64>(dy.data.vz.data());
            auto* im = std::assume_aligned<64>(dy.data.inv_mass.data());
            auto* px_prev = std::assume_aligned<64>(dy.prev_x.data());
            auto* py_prev = std::assume_aligned<64>(dy.prev_y.data());
            auto* pz_prev = std::assume_aligned<64>(dy.prev_z.data());
            for (usize i = r.begin(); i != r.end(); ++i) {
                if (im[i] > 0.0f) {
                    f32 vx_i = (px[i] - px_prev[i]) / dt;
                    f32 vy_i = (py[i] - py_prev[i]) / dt;
                    f32 vz_i = (pz[i] - pz_prev[i]) / dt;
                    vx[i] = vx_i * k; vy[i] = vy_i * k; vz[i] = vz_i * k;
                } else {
                    vx[i] = vy[i] = vz[i] = 0.0f;
                    px[i] = px_prev[i]; py[i] = py_prev[i]; pz[i] = pz_prev[i];
                }
            }
        }, part);
    }
};

ISim* make_tbb(const InitDesc& desc) {
    auto* s = new TbbSim();
    s->init(desc);
    return s;
}

#else // HINAPE_HAVE_TBB

ISim* make_tbb(const InitDesc& desc) {
    return make_native(desc);
}

#endif

} // namespace detail
} // namespace HinaPE
