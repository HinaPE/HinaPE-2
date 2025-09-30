#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "cloth.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace HinaPE;
using Catch::Matchers::WithinAbs;

static void make_grid(int nx, int ny, float spacing, std::vector<float>& xyz, std::vector<u32>& tris, std::vector<u32>& fixed) {
    int n = nx * ny;
    xyz.resize(n * 3);
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            int id = y * nx + x;
            xyz[id*3+0] = x * spacing;
            xyz[id*3+1] = 0.0f;
            xyz[id*3+2] = y * spacing;
        }
    }
    fixed.clear();
    for (int x = 0; x < nx; ++x) fixed.push_back(static_cast<u32>(x));
    tris.clear();
    for (int y = 0; y < ny-1; ++y) {
        for (int x = 0; x < nx-1; ++x) {
            int a = y*nx + x;
            int b = y*nx + x+1;
            int c = (y+1)*nx + x;
            int d = (y+1)*nx + x+1;
            tris.push_back(static_cast<u32>(a));
            tris.push_back(static_cast<u32>(b));
            tris.push_back(static_cast<u32>(d));
            tris.push_back(static_cast<u32>(a));
            tris.push_back(static_cast<u32>(d));
            tris.push_back(static_cast<u32>(c));
        }
    }
}

static void build_edges_from_triangles(const std::vector<u32>& tris, std::vector<std::pair<u32,u32>>& edges) {
    size_t m = tris.size() / 3;
    edges.clear();
    edges.reserve(m * 3);
    for (size_t t = 0; t < m; ++t) {
        u32 a = tris[t*3+0];
        u32 b = tris[t*3+1];
        u32 c = tris[t*3+2];
        u32 e0a = a < b ? a : b; u32 e0b = a < b ? b : a;
        u32 e1a = b < c ? b : c; u32 e1b = b < c ? c : b;
        u32 e2a = c < a ? c : a; u32 e2b = c < a ? a : c;
        edges.emplace_back(e0a, e0b);
        edges.emplace_back(e1a, e1b);
        edges.emplace_back(e2a, e2b);
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
}

static double rms_edge_error(const DynamicView& v, const std::vector<std::pair<u32,u32>>& edges, const std::vector<float>& rest) {
    double acc = 0.0;
    size_t m = edges.size();
    for (size_t k = 0; k < m; ++k) {
        u32 i = edges[k].first;
        u32 j = edges[k].second;
        double dx = double(v.pos_x[i]) - double(v.pos_x[j]);
        double dy = double(v.pos_y[i]) - double(v.pos_y[j]);
        double dz = double(v.pos_z[i]) - double(v.pos_z[j]);
        double len = std::sqrt(dx*dx + dy*dy + dz*dz);
        double e = len - double(rest[k]);
        acc += e * e;
    }
    return std::sqrt(acc / double(std::max<size_t>(1, m)));
}

static void compute_rest_from_xyz(const std::vector<float>& xyz, const std::vector<std::pair<u32,u32>>& edges, std::vector<float>& rest) {
    rest.resize(edges.size());
    for (size_t k = 0; k < edges.size(); ++k) {
        u32 i = edges[k].first;
        u32 j = edges[k].second;
        float dx = xyz[i*3+0] - xyz[j*3+0];
        float dy = xyz[i*3+1] - xyz[j*3+1];
        float dz = xyz[i*3+2] - xyz[j*3+2];
        rest[k] = std::sqrt(dx*dx + dy*dy + dz*dz);
    }
}

TEST_CASE("create_destroy_and_map") {
    std::vector<float> xyz;
    std::vector<u32> tris;
    std::vector<u32> fixed;
    make_grid(8, 8, 0.1f, xyz, tris, fixed);
    ExecPolicy exec{};
    SolvePolicy solve{};
    InitDesc init{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), exec, solve};
    Handle h = create(init);
    REQUIRE(h != nullptr);
    DynamicView v = map_dynamic(h);
    REQUIRE(v.count == xyz.size()/3);
    REQUIRE(v.pos_x != nullptr);
    REQUIRE(v.vel_y != nullptr);
    StepParams sp{};
    step(h, sp);
    destroy(h);
}

TEST_CASE("fixed_points_hold") {
    std::vector<float> xyz;
    std::vector<u32> tris;
    std::vector<u32> fixed;
    int nx = 16;
    int ny = 16;
    make_grid(nx, ny, 0.05f, xyz, tris, fixed);
    ExecPolicy exec{};
    SolvePolicy solve{};
    solve.iterations = 10;
    InitDesc init{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), exec, solve};
    Handle h = create(init);
    DynamicView v = map_dynamic(h);
    StepParams sp{};
    for (int i = 0; i < 60; ++i) step(h, sp);
    for (int x = 0; x < nx; ++x) {
        size_t id = static_cast<size_t>(x);
        CHECK_THAT(v.pos_x[id], WithinAbs(x * 0.05f, 1e-6f));
        CHECK_THAT(v.pos_y[id], WithinAbs(0.0f, 1e-6f));
        CHECK_THAT(v.pos_z[id], WithinAbs(0.0f, 1e-6f));
    }
    destroy(h);
}

TEST_CASE("gravity_effect") {
    std::vector<float> xyz;
    std::vector<u32> tris;
    std::vector<u32> fixed;
    int nx = 8;
    int ny = 8;
    make_grid(nx, ny, 0.1f, xyz, tris, fixed);
    ExecPolicy exec{};
    SolvePolicy solve{};
    InitDesc init{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), exec, solve};
    Handle h = create(init);
    DynamicView v = map_dynamic(h);
    StepParams sp{};
    double sumy0 = 0.0;
    for (size_t i = 0; i < v.count; ++i) sumy0 += v.pos_y[i];
    for (int i = 0; i < 120; ++i) step(h, sp);
    double sumy1 = 0.0;
    for (size_t i = 0; i < v.count; ++i) sumy1 += v.pos_y[i];
    CHECK(sumy1 < sumy0 - 1e-6);
    destroy(h);
}

TEST_CASE("constraint_convergence_from_perturbation") {
    std::vector<float> xyz;
    std::vector<u32> tris;
    std::vector<u32> fixed;
    int nx = 16;
    int ny = 16;
    make_grid(nx, ny, 0.05f, xyz, tris, fixed);
    std::vector<std::pair<u32,u32>> edges;
    build_edges_from_triangles(tris, edges);
    std::vector<float> rest;
    compute_rest_from_xyz(xyz, edges, rest);
    ExecPolicy exec{};
    SolvePolicy solve{};
    solve.iterations = 20;
    solve.substeps = 1;
    solve.compliance_stretch = 0.0f;
    InitDesc init{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), exec, solve};
    Handle h = create(init);
    DynamicView v = map_dynamic(h);
    for (size_t i = 0; i < v.count; ++i) if (std::none_of(fixed.begin(), fixed.end(), [&](u32 f){return f==i;})) v.pos_y[i] += 0.05f;
    StepParams sp{};
    sp.gravity_x = 0.0f;
    sp.gravity_y = 0.0f;
    sp.gravity_z = 0.0f;
    double e0 = rms_edge_error(v, edges, rest);
    for (int i = 0; i < 10; ++i) step(h, sp);
    double e1 = rms_edge_error(v, edges, rest);
    CHECK(e1 < e0 * 0.5);
    destroy(h);
}

TEST_CASE("compliance_effect") {
    std::vector<float> xyz;
    std::vector<u32> tris;
    std::vector<u32> fixed;
    int nx = 16;
    int ny = 16;
    make_grid(nx, ny, 0.05f, xyz, tris, fixed);
    std::vector<std::pair<u32,u32>> edges;
    build_edges_from_triangles(tris, edges);
    std::vector<float> rest;
    compute_rest_from_xyz(xyz, edges, rest);
    ExecPolicy exec{};
    SolvePolicy rigid{};
    rigid.iterations = 15;
    rigid.compliance_stretch = 0.0f;
    SolvePolicy soft = rigid;
    soft.compliance_stretch = 1e-2f;
    InitDesc inita{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), exec, rigid};
    InitDesc initb{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), exec, soft};
    Handle ha = create(inita);
    Handle hb = create(initb);
    DynamicView va = map_dynamic(ha);
    DynamicView vb = map_dynamic(hb);
    for (size_t i = 0; i < va.count; ++i) if (std::none_of(fixed.begin(), fixed.end(), [&](u32 f){return f==i;})) { va.pos_y[i] += 0.03f; vb.pos_y[i] += 0.03f; }
    StepParams sp{};
    sp.gravity_x = 0.0f;
    sp.gravity_y = 0.0f;
    sp.gravity_z = 0.0f;
    for (int i = 0; i < 8; ++i) { step(ha, sp); step(hb, sp); }
    double ea = rms_edge_error(va, edges, rest);
    double eb = rms_edge_error(vb, edges, rest);
    CHECK(ea < eb);
    destroy(ha);
    destroy(hb);
}

TEST_CASE("determinism_same_setup") {
    std::vector<float> xyz;
    std::vector<u32> tris;
    std::vector<u32> fixed;
    int nx = 12;
    int ny = 12;
    make_grid(nx, ny, 0.07f, xyz, tris, fixed);
    ExecPolicy exec{};
    SolvePolicy solve{};
    solve.iterations = 12;
    solve.substeps = 2;
    InitDesc inita{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), exec, solve};
    InitDesc initb = inita;
    Handle ha = create(inita);
    Handle hb = create(initb);
    DynamicView va = map_dynamic(ha);
    DynamicView vb = map_dynamic(hb);
    StepParams sp{};
    for (int i = 0; i < 120; ++i) { step(ha, sp); step(hb, sp); }
    for (size_t i = 0; i < va.count; ++i) {
        CHECK_THAT(va.pos_x[i], WithinAbs(vb.pos_x[i], 1e-7f));
        CHECK_THAT(va.pos_y[i], WithinAbs(vb.pos_y[i], 1e-7f));
        CHECK_THAT(va.pos_z[i], WithinAbs(vb.pos_z[i], 1e-7f));
    }
    destroy(ha);
    destroy(hb);
}

TEST_CASE("damping_reduces_speed") {
    std::vector<float> xyz;
    std::vector<u32> tris;
    std::vector<u32> fixed;
    int nx = 10;
    int ny = 10;
    make_grid(nx, ny, 0.06f, xyz, tris, fixed);
    ExecPolicy exec{};
    SolvePolicy solve{};
    solve.damping = 0.2f;
    solve.iterations = 1;
    InitDesc init{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), exec, solve};
    Handle h = create(init);
    DynamicView v = map_dynamic(h);
    for (size_t i = 0; i < v.count; ++i) { v.vel_y[i] = 1.0f; }
    StepParams sp{};
    sp.gravity_x = 0.0f;
    sp.gravity_y = 0.0f;
    sp.gravity_z = 0.0f;
    double ke0 = 0.0;
    for (size_t i = 0; i < v.count; ++i) { double vx = v.vel_x[i]; double vy = v.vel_y[i]; double vz = v.vel_z[i]; ke0 += 0.5*(vx*vx+vy*vy+vz*vz); }
    step(h, sp);
    double ke1 = 0.0;
    for (size_t i = 0; i < v.count; ++i) { double vx = v.vel_x[i]; double vy = v.vel_y[i]; double vz = v.vel_z[i]; ke1 += 0.5*(vx*vx+vy*vy+vz*vz); }
    CHECK(ke1 < ke0);
    destroy(h);
}

TEST_CASE("zero_dt_fallback") {
    std::vector<float> xyz;
    std::vector<u32> tris;
    std::vector<u32> fixed;
    make_grid(6, 6, 0.1f, xyz, tris, fixed);
    ExecPolicy exec{};
    SolvePolicy solve{};
    InitDesc init{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), exec, solve};
    Handle h = create(init);
    DynamicView v = map_dynamic(h);
    StepParams sp{};
    sp.dt = 0.0f;
    step(h, sp);
    CHECK(std::isfinite(v.pos_y[0]));
    destroy(h);
}

TEST_CASE("backend_equivalence_native_vs_avx2") {
    std::vector<float> xyz;
    std::vector<u32> tris;
    std::vector<u32> fixed;
    int nx = 10;
    int ny = 10;
    make_grid(nx, ny, 0.05f, xyz, tris, fixed);
    SolvePolicy solve{}; // default: compliance 0, deterministic
    solve.iterations = 6;
    solve.substeps = 2;
    ExecPolicy exec_native{}; exec_native.backend = ExecPolicy::Backend::Native;
    ExecPolicy exec_avx2{};   exec_avx2.backend   = ExecPolicy::Backend::Avx2;
    InitDesc init_native{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), exec_native, solve};
    InitDesc init_avx2  {std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), exec_avx2, solve};
    Handle h_native = create(init_native);
    Handle h_avx2   = create(init_avx2);
    REQUIRE(h_native != nullptr);
    REQUIRE(h_avx2 != nullptr);
    DynamicView vN = map_dynamic(h_native);
    DynamicView vA = map_dynamic(h_avx2);
    StepParams sp{};
    for (int i = 0; i < 60; ++i) { step(h_native, sp); step(h_avx2, sp); }
    for (size_t i = 0; i < vN.count; ++i) {
        CHECK_THAT(vN.pos_x[i], WithinAbs(vA.pos_x[i], 1e-5f));
        CHECK_THAT(vN.pos_y[i], WithinAbs(vA.pos_y[i], 1e-5f));
        CHECK_THAT(vN.pos_z[i], WithinAbs(vA.pos_z[i], 1e-5f));
    }
    destroy(h_native);
    destroy(h_avx2);
}
