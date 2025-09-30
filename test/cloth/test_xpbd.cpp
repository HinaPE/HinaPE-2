
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "cloth.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace HinaPE;
using Catch::Matchers::WithinAbs;

namespace {
    void make_grid(int nx, int ny, float spacing, std::vector<float>& xyz, std::vector<u32>& tris, std::vector<u32>& fixed) {
        const int n = nx * ny;
        xyz.resize(n * 3);
        tris.clear();
        fixed.clear();
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                const int id = y * nx + x;
                xyz[id * 3 + 0] = x * spacing;
                xyz[id * 3 + 1] = 0.0f;
                xyz[id * 3 + 2] = y * spacing;
            }
        }
        for (int x = 0; x < nx; ++x) fixed.push_back(static_cast<u32>(x));
        for (int y = 0; y < ny - 1; ++y) {
            for (int x = 0; x < nx - 1; ++x) {
                const int a = y * nx + x;
                const int b = y * nx + x + 1;
                const int c = (y + 1) * nx + x;
                const int d = (y + 1) * nx + x + 1;
                tris.push_back(static_cast<u32>(a)); tris.push_back(static_cast<u32>(b)); tris.push_back(static_cast<u32>(d));
                tris.push_back(static_cast<u32>(a)); tris.push_back(static_cast<u32>(d)); tris.push_back(static_cast<u32>(c));
            }
        }
    }

    void build_edges(const std::vector<u32>& tris, std::vector<std::pair<u32, u32>>& edges) {
        const size_t m = tris.size() / 3;
        edges.clear(); edges.reserve(m * 3);
        for (size_t t = 0; t < m; ++t) {
            const u32 a = tris[t * 3 + 0];
            const u32 b = tris[t * 3 + 1];
            const u32 c = tris[t * 3 + 2];
            const auto add = [&](u32 i, u32 j) {
                if (i > j) std::swap(i, j);
                edges.emplace_back(i, j);
            };
            add(a, b); add(b, c); add(c, a);
        }
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    }

    void compute_rest(const std::vector<float>& xyz, const std::vector<std::pair<u32, u32>>& edges, std::vector<float>& rest) {
        rest.resize(edges.size());
        for (size_t k = 0; k < edges.size(); ++k) {
            const u32 i = edges[k].first; const u32 j = edges[k].second;
            const float dx = xyz[i * 3 + 0] - xyz[j * 3 + 0];
            const float dy = xyz[i * 3 + 1] - xyz[j * 3 + 1];
            const float dz = xyz[i * 3 + 2] - xyz[j * 3 + 2];
            rest[k] = std::sqrt(dx * dx + dy * dy + dz * dz);
        }
    }

    double rms_edge_error(const DynamicView& v, const std::vector<std::pair<u32, u32>>& edges, const std::vector<float>& rest) {
        double acc = 0.0; const size_t m = edges.size();
        for (size_t k = 0; k < m; ++k) {
            const u32 i = edges[k].first, j = edges[k].second;
            const double dx = double(v.pos_x[i]) - double(v.pos_x[j]);
            const double dy = double(v.pos_y[i]) - double(v.pos_y[j]);
            const double dz = double(v.pos_z[i]) - double(v.pos_z[j]);
            const double len = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double e   = len - double(rest[k]);
            acc += e * e;
        }
        return std::sqrt(acc / double(std::max<size_t>(1, m)));
    }
}

TEST_CASE("api_create_destroy_step_map") {
    std::vector<float> xyz; std::vector<u32> tris; std::vector<u32> fixed;
    make_grid(8, 8, 0.1f, xyz, tris, fixed);
    ExecPolicy ex{}; SolvePolicy sv{}; Handle h = create(InitDesc{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), ex, sv});
    REQUIRE(h != nullptr);
    DynamicView v = map_dynamic(h);
    REQUIRE(v.count == xyz.size() / 3);
    REQUIRE(v.pos_x != nullptr); REQUIRE(v.vel_y != nullptr);
    step(h, StepParams{});
    destroy(h);
}

TEST_CASE("fixed_vertices_remain_pinned") {
    std::vector<float> xyz; std::vector<u32> tris; std::vector<u32> fixed;
    const int nx = 16, ny = 16; make_grid(nx, ny, 0.05f, xyz, tris, fixed);
    ExecPolicy ex{}; SolvePolicy sv{}; sv.iterations=10; auto h = create(InitDesc{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), ex, sv});
    auto v = map_dynamic(h);
    for (int i = 0; i < 60; ++i) step(h, StepParams{});
    for (int x = 0; x < nx; ++x) {
        const size_t id = static_cast<size_t>(x);
        CHECK_THAT(v.pos_x[id], WithinAbs(x * 0.05f, 1e-6f));
        CHECK_THAT(v.pos_y[id], WithinAbs(0.0f, 1e-6f));
        CHECK_THAT(v.pos_z[id], WithinAbs(0.0f, 1e-6f));
    }
    destroy(h);
}

TEST_CASE("gravity_moves_center_of_mass_down") {
    std::vector<float> xyz; std::vector<u32> tris; std::vector<u32> fixed;
    make_grid(8, 8, 0.1f, xyz, tris, fixed);
    ExecPolicy ex{}; SolvePolicy sv{}; auto h = create(InitDesc{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), ex, sv});
    auto v = map_dynamic(h);
    double sumy0 = std::accumulate(v.pos_y, v.pos_y + v.count, 0.0);
    for (int i = 0; i < 120; ++i) step(h, StepParams{});
    double sumy1 = std::accumulate(v.pos_y, v.pos_y + v.count, 0.0);
    CHECK(sumy1 < sumy0 - 1e-6);
    destroy(h);
}

TEST_CASE("constraints_converge_from_perturbation") {
    std::vector<float> xyz; std::vector<u32> tris; std::vector<u32> fixed;
    const int nx = 16, ny = 16; make_grid(nx, ny, 0.05f, xyz, tris, fixed);
    std::vector<std::pair<u32, u32>> edges; build_edges(tris, edges);
    std::vector<float> rest; compute_rest(xyz, edges, rest);
    ExecPolicy ex{}; SolvePolicy sv{}; sv.iterations=20; sv.substeps=1; sv.compliance_stretch=0.0f; auto h = create(InitDesc{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), ex, sv});
    auto v = map_dynamic(h);
    for (size_t i = 0; i < v.count; ++i) if (std::find(fixed.begin(), fixed.end(), static_cast<u32>(i)) == fixed.end()) v.pos_y[i] += 0.05f;
    StepParams sp{}; sp.gravity_x = sp.gravity_y = sp.gravity_z = 0.0f;
    const double e0 = rms_edge_error(v, edges, rest);
    for (int i = 0; i < 10; ++i) step(h, sp);
    const double e1 = rms_edge_error(v, edges, rest);
    CHECK(e1 < e0 * 0.5);
    destroy(h);
}

TEST_CASE("compliance_softens_solution") {
    std::vector<float> xyz; std::vector<u32> tris; std::vector<u32> fixed;
    const int nx = 16, ny = 16; make_grid(nx, ny, 0.05f, xyz, tris, fixed);
    std::vector<std::pair<u32, u32>> edges; build_edges(tris, edges);
    std::vector<float> rest; compute_rest(xyz, edges, rest);
    SolvePolicy rigid{}; rigid.iterations=15; rigid.compliance_stretch=0.0f;
    SolvePolicy soft = rigid; soft.compliance_stretch = 1e-2f;
    auto ha = create(InitDesc{xyz, tris, fixed, ExecPolicy{}, rigid});
    auto hb = create(InitDesc{xyz, tris, fixed, ExecPolicy{}, soft});
    auto va = map_dynamic(ha); auto vb = map_dynamic(hb);
    for (size_t i = 0; i < va.count; ++i) if (std::find(fixed.begin(), fixed.end(), static_cast<u32>(i)) == fixed.end()) { va.pos_y[i] += 0.03f; vb.pos_y[i] += 0.03f; }
    StepParams sp{}; sp.gravity_x = sp.gravity_y = sp.gravity_z = 0.0f;
    for (int i = 0; i < 8; ++i) { step(ha, sp); step(hb, sp); }
    const double ea = rms_edge_error(va, edges, rest); const double eb = rms_edge_error(vb, edges, rest);
    CHECK(ea < eb);
    destroy(ha); destroy(hb);
}

TEST_CASE("determinism_same_setup") {
    std::vector<float> xyz; std::vector<u32> tris; std::vector<u32> fixed; make_grid(12, 12, 0.07f, xyz, tris, fixed);
    SolvePolicy solve{}; solve.iterations=12; solve.substeps=2;
    auto ha = create(InitDesc{xyz, tris, fixed, ExecPolicy{}, solve});
    auto hb = create(InitDesc{xyz, tris, fixed, ExecPolicy{}, solve});
    auto va = map_dynamic(ha); auto vb = map_dynamic(hb);
    for (int i = 0; i < 120; ++i) { step(ha, StepParams{}); step(hb, StepParams{}); }
    for (size_t i = 0; i < va.count; ++i) {
        CHECK_THAT(va.pos_x[i], WithinAbs(vb.pos_x[i], 1e-7f));
        CHECK_THAT(va.pos_y[i], WithinAbs(vb.pos_y[i], 1e-7f));
        CHECK_THAT(va.pos_z[i], WithinAbs(vb.pos_z[i], 1e-7f));
    }
    destroy(ha); destroy(hb);
}

TEST_CASE("damping_reduces_kinetic_energy") {
    std::vector<float> xyz; std::vector<u32> tris; std::vector<u32> fixed; make_grid(10, 10, 0.06f, xyz, tris, fixed);
    SolvePolicy solve{}; solve.iterations=1; solve.damping=0.2f;
    auto h = create(InitDesc{xyz, tris, fixed, ExecPolicy{}, solve}); auto v = map_dynamic(h);
    for (size_t i = 0; i < v.count; ++i) v.vel_y[i] = 1.0f;
    StepParams sp{}; sp.gravity_x = sp.gravity_y = sp.gravity_z = 0.0f;
    auto kinetic = [&](const DynamicView& dv) {
        double k = 0.0; for (size_t i = 0; i < dv.count; ++i) { const double vx = dv.vel_x[i], vy = dv.vel_y[i], vz = dv.vel_z[i]; k += 0.5 * (vx*vx + vy*vy + vz*vz);} return k; };
    const double ke0 = kinetic(v); step(h, sp); const double ke1 = kinetic(v);
    CHECK(ke1 < ke0);
    destroy(h);
}

TEST_CASE("zero_dt_is_safe") {
    std::vector<float> xyz; std::vector<u32> tris; std::vector<u32> fixed; make_grid(6, 6, 0.1f, xyz, tris, fixed);
    ExecPolicy ex{}; SolvePolicy sv{}; auto h = create(InitDesc{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), ex, sv}); auto v = map_dynamic(h);
    StepParams sp{}; sp.dt = 0.0f; step(h, sp); CHECK(std::isfinite(v.pos_y[0])); destroy(h);
}

TEST_CASE("backend_request_simd_matches_native") {
    std::vector<float> xyz; std::vector<u32> tris; std::vector<u32> fixed; make_grid(10, 10, 0.05f, xyz, tris, fixed);
    SolvePolicy solve{}; solve.iterations=6; solve.substeps=2;
    ExecPolicy exN{}; exN.backend = ExecPolicy::Backend::Native; ExecPolicy exA{}; exA.backend = ExecPolicy::Backend::Simd;
    auto hN = create(InitDesc{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), exN, solve}); auto hA = create(InitDesc{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), exA, solve});
    REQUIRE(hN != nullptr); REQUIRE(hA != nullptr);
    auto vN = map_dynamic(hN); auto vA = map_dynamic(hA);
    for (int i = 0; i < 60; ++i) { step(hN, StepParams{}); step(hA, StepParams{}); }
    for (size_t i = 0; i < vN.count; ++i) {
        CHECK_THAT(vN.pos_x[i], WithinAbs(vA.pos_x[i], 1e-5f));
        CHECK_THAT(vN.pos_y[i], WithinAbs(vA.pos_y[i], 1e-5f));
        CHECK_THAT(vN.pos_z[i], WithinAbs(vA.pos_z[i], 1e-5f));
    }
    destroy(hN); destroy(hA);
}

TEST_CASE("backend_request_tbb_matches_native") {
    std::vector<float> xyz; std::vector<u32> tris; std::vector<u32> fixed; make_grid(10, 10, 0.05f, xyz, tris, fixed);
    SolvePolicy solve{}; solve.iterations=6; solve.substeps=2;
    ExecPolicy exN{}; exN.backend = ExecPolicy::Backend::Native; ExecPolicy exT{}; exT.backend = ExecPolicy::Backend::Tbb;
    auto hN = create(InitDesc{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), exN, solve});
    auto hT = create(InitDesc{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), exT, solve});
    REQUIRE(hN != nullptr); REQUIRE(hT != nullptr);
    auto vN = map_dynamic(hN); auto vT = map_dynamic(hT);
    for (int i = 0; i < 60; ++i) { step(hN, StepParams{}); step(hT, StepParams{}); }
    for (size_t i = 0; i < vN.count; ++i) {
        CHECK_THAT(vN.pos_x[i], WithinAbs(vT.pos_x[i], 1e-5f));
        CHECK_THAT(vN.pos_y[i], WithinAbs(vT.pos_y[i], 1e-5f));
        CHECK_THAT(vN.pos_z[i], WithinAbs(vT.pos_z[i], 1e-5f));
    }
    destroy(hN); destroy(hT);
}

TEST_CASE("determinism_same_setup_tbb") {
    std::vector<float> xyz; std::vector<u32> tris; std::vector<u32> fixed; make_grid(12, 12, 0.07f, xyz, tris, fixed);
    SolvePolicy solve{}; solve.iterations=12; solve.substeps=2;
    ExecPolicy ex{}; ex.backend = ExecPolicy::Backend::Tbb;
    auto ha = create(InitDesc{xyz, tris, fixed, ex, solve});
    auto hb = create(InitDesc{xyz, tris, fixed, ex, solve});
    REQUIRE(ha != nullptr); REQUIRE(hb != nullptr);
    auto va = map_dynamic(ha); auto vb = map_dynamic(hb);
    for (int i = 0; i < 120; ++i) { step(ha, StepParams{}); step(hb, StepParams{}); }
    for (size_t i = 0; i < va.count; ++i) {
        CHECK_THAT(va.pos_x[i], WithinAbs(vb.pos_x[i], 1e-7f));
        CHECK_THAT(va.pos_y[i], WithinAbs(vb.pos_y[i], 1e-7f));
        CHECK_THAT(va.pos_z[i], WithinAbs(vb.pos_z[i], 1e-7f));
    }
    destroy(ha); destroy(hb);
}

TEST_CASE("tbb_threads_variants_match") {
    std::vector<float> xyz; std::vector<u32> tris; std::vector<u32> fixed; make_grid(10, 10, 0.05f, xyz, tris, fixed);
    SolvePolicy solve{}; solve.iterations=8; solve.substeps=2;
    ExecPolicy exA{}; exA.backend = ExecPolicy::Backend::Tbb; exA.threads = 0; // implementation decides
    ExecPolicy exB{}; exB.backend = ExecPolicy::Backend::Tbb; exB.threads = 2; // limit parallelism explicitly
    auto hA = create(InitDesc{xyz, tris, fixed, exA, solve});
    auto hB = create(InitDesc{xyz, tris, fixed, exB, solve});
    REQUIRE(hA != nullptr); REQUIRE(hB != nullptr);
    auto vA = map_dynamic(hA); auto vB = map_dynamic(hB);
    for (int i = 0; i < 80; ++i) { step(hA, StepParams{}); step(hB, StepParams{}); }
    for (size_t i = 0; i < vA.count; ++i) {
        CHECK_THAT(vA.pos_x[i], WithinAbs(vB.pos_x[i], 1e-5f));
        CHECK_THAT(vA.pos_y[i], WithinAbs(vB.pos_y[i], 1e-5f));
        CHECK_THAT(vA.pos_z[i], WithinAbs(vB.pos_z[i], 1e-5f));
    }
    destroy(hA); destroy(hB);
}


