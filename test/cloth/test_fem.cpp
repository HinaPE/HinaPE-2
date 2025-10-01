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
}

TEST_CASE("fem_api_create_destroy_step_map") {
    std::vector<float> xyz; std::vector<u32> tris; std::vector<u32> fixed;
    make_grid(8, 8, 0.1f, xyz, tris, fixed);
    ExecPolicy ex{}; ex.backend = ExecPolicy::Backend::Fem; SolvePolicy sv{}; Handle h = create(InitDesc{xyz, tris, fixed, ex, sv});
    REQUIRE(h != nullptr);
    DynamicView v = map_dynamic(h);
    REQUIRE(v.count == xyz.size() / 3);
    step(h, StepParams{});
    destroy(h);
}

TEST_CASE("fem_fixed_vertices_remain_pinned") {
    std::vector<float> xyz; std::vector<u32> tris; std::vector<u32> fixed;
    const int nx = 16, ny = 16; make_grid(nx, ny, 0.05f, xyz, tris, fixed);
    ExecPolicy ex{}; ex.backend = ExecPolicy::Backend::Fem; SolvePolicy sv{}; sv.iterations=30; auto h = create(InitDesc{xyz, tris, fixed, ex, sv});
    auto v = map_dynamic(h);
    for (int i = 0; i < 40; ++i) step(h, StepParams{});
    for (int x = 0; x < nx; ++x) {
        const size_t id = static_cast<size_t>(x);
        CHECK_THAT(v.pos_x[id], WithinAbs(x * 0.05f, 1e-4f));
        CHECK_THAT(v.pos_y[id], WithinAbs(0.0f, 1e-4f));
        CHECK_THAT(v.pos_z[id], WithinAbs(0.0f, 1e-4f));
    }
    destroy(h);
}

TEST_CASE("fem_gravity_moves_com_down") {
    std::vector<float> xyz; std::vector<u32> tris; std::vector<u32> fixed;
    make_grid(8, 8, 0.1f, xyz, tris, fixed);
    ExecPolicy ex{}; ex.backend = ExecPolicy::Backend::Fem; SolvePolicy sv{}; auto h = create(InitDesc{xyz, tris, fixed, ex, sv});
    auto v = map_dynamic(h);
    double sumy0 = std::accumulate(v.pos_y, v.pos_y + v.count, 0.0);
    for (int i = 0; i < 60; ++i) step(h, StepParams{});
    double sumy1 = std::accumulate(v.pos_y, v.pos_y + v.count, 0.0);
    CHECK(sumy1 < sumy0 - 1e-6);
    destroy(h);
}

TEST_CASE("fem_zero_dt_is_safe") {
    std::vector<float> xyz; std::vector<u32> tris; std::vector<u32> fixed; make_grid(6, 6, 0.1f, xyz, tris, fixed);
    ExecPolicy ex{}; ex.backend = ExecPolicy::Backend::Fem; SolvePolicy sv{}; auto h = create(InitDesc{xyz, tris, fixed, ex, sv});
    auto v = map_dynamic(h);
    StepParams sp{}; sp.dt = 0.0f; step(h, sp);
    CHECK(std::isfinite(v.pos_y[0]));
    destroy(h);
}

