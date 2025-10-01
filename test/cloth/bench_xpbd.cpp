#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "cloth.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace HinaPE;

namespace {
// Regular grid mesh builder (positions, triangle indices, and fixed vertices)
void make_grid(int nx, int ny, float spacing, std::vector<float>& xyz, std::vector<u32>& tris, std::vector<u32>& fixed) {
    const int n = nx * ny;
    xyz.resize(static_cast<size_t>(n) * 3);
    tris.clear();
    fixed.clear();
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const int id      = y * nx + x;
            xyz[id * 3 + 0]   = x * spacing;
            xyz[id * 3 + 1]   = 0.0f;
            xyz[id * 3 + 2]   = y * spacing;
        }
    }
    // Pin the top edge (y == 0)
    for (int x = 0; x < nx; ++x) fixed.push_back(static_cast<u32>(x));
    // Two triangles per grid quad
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

// Unique undirected edge list for approximate constraint count
void build_edges(const std::vector<u32>& tris, std::vector<std::pair<u32, u32>>& edges) {
    const size_t m = tris.size() / 3;
    edges.clear();
    edges.reserve(m * 3);
    for (size_t t = 0; t < m; ++t) {
        u32 a = tris[t * 3 + 0];
        u32 b = tris[t * 3 + 1];
        u32 c = tris[t * 3 + 2];
        auto add = [&](u32 i, u32 j) {
            if (i > j) std::swap(i, j);
            edges.emplace_back(i, j);
        };
        add(a, b); add(b, c); add(c, a);
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
}

std::string backend_name(ExecPolicy::Backend b) {
    switch (b) {
        case ExecPolicy::Backend::Native: return "native";
        case ExecPolicy::Backend::Simd:   return "simd";
        case ExecPolicy::Backend::Tbb:    return "tbb";
    }
    return "unknown";
}
}

TEST_CASE("xpbd_backend_performance_sweep", "[xpbd][bench]") {
    // Note: Catch2 benchmarking CLI options can further refine statistical rigor, e.g.:
    //   --benchmark-samples 100 --benchmark-warmup-time 0.5 --benchmark-confidence-interval 0.95
    // These defaults are acceptable for CI; increase locally for paper-level reporting.

    const int sizes[] = {32, 64, 128}; // problem sizes (grid is size x size)
    const int solver_iterations[] = {8, 16}; // representative iteration counts
    const int substeps = 1;
    const float spacing = 0.05f;
    const int steps_per_measure = 64; // number of simulation steps within the measured region

    // Common simulation parameters
    StepParams sp{}; // dt = 1/60 by default; gravity enabled
    SolvePolicy solve{}; // baseline solver config
    solve.substeps = substeps;

    // Inform about available backends at runtime (printed once per test case)
#if defined(HINAPE_HAVE_SIMD)
    std::cout << "[bench] SIMD backend: available" << std::endl;
#else
    std::cout << "[bench] SIMD backend: not available (falls back to native)" << std::endl;
#endif
#if defined(HINAPE_HAVE_TBB)
    std::cout << "[bench] TBB backend: available" << std::endl;
#else
    std::cout << "[bench] TBB backend: not available (falls back to native)" << std::endl;
#endif

    for (int n : sizes) {
        DYNAMIC_SECTION("grid=" << n << "x" << n) {
            std::vector<float> xyz;
            std::vector<u32> tris;
            std::vector<u32> fixed;
            make_grid(n, n, spacing, xyz, tris, fixed);
            std::vector<std::pair<u32, u32>> edges;
            build_edges(tris, edges);

            for (int iters : solver_iterations) {
                DYNAMIC_SECTION("iters=" << iters) {
                    solve.iterations = iters;
                    solve.damping = 0.01f;
                    solve.compliance_stretch = 0.0f;

                    // Benchmark each backend under identical physical parameters
                    for (ExecPolicy::Backend b : {ExecPolicy::Backend::Native, ExecPolicy::Backend::Simd, ExecPolicy::Backend::Tbb}) {
                        ExecPolicy ex{}; ex.backend = b; ex.threads = 0; ex.deterministic = true; ex.telemetry = false;

                        const std::string label = std::string("backend=") + backend_name(b) +
                            ", verts=" + std::to_string(n * n) +
                            ", edges~=" + std::to_string(edges.size()) +
                            ", steps=" + std::to_string(steps_per_measure);

                        BENCHMARK_ADVANCED(label.c_str())
                        (Catch::Benchmark::Chronometer meter) {
                            // Create the simulation outside the measured region to exclude setup overhead
                            Handle h = create(InitDesc{std::span<const float>(xyz.data(), xyz.size()),
                                                       std::span<const u32>(tris.data(), tris.size()),
                                                       std::span<const u32>(fixed.data(), fixed.size()),
                                                       ex, solve});
                            REQUIRE(h != nullptr);
                            auto v = map_dynamic(h);
                            // Optional: tiny warmup to reach a steady state
                            for (int w = 0; w < 8; ++w) step(h, sp);

                            // Measure a fixed number of steps. Catch2 will repeat this block to collect samples.
                            meter.measure([&] {
                                for (int i = 0; i < steps_per_measure; ++i) {
                                    step(h, sp);
                                }
                            });

                            // Cleanup outside measurement
                            destroy(h);
                        };
                    }
                }
            }
        }
    }
}

TEST_CASE("xpbd_tbb_thread_scaling", "[xpbd][bench][tbb]") {
    // Characterize parallel scaling when the TBB backend is present. If TBB is
    // unavailable, this degenerates to native and still runs for completeness.
    const int n = 128;             // moderate problem size
    const int steps_per_measure = 64;
    const float spacing = 0.05f;
    const int iters = 16;
    const int substeps = 1;

    std::vector<float> xyz; std::vector<u32> tris; std::vector<u32> fixed;
    make_grid(n, n, spacing, xyz, tris, fixed);
    std::vector<std::pair<u32, u32>> edges; build_edges(tris, edges);

    SolvePolicy solve{}; solve.iterations = iters; solve.substeps = substeps; solve.damping = 0.01f; solve.compliance_stretch = 0.0f;
    StepParams sp{};

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int thread_settings[] = {0, 2, 4, 8};
    for (int th : thread_settings) {
        // Skip obviously excessive thread counts relative to hardware to reduce noise
        if (th > 0 && static_cast<unsigned>(th) > 2 * hw) continue;

        ExecPolicy ex{}; ex.backend = ExecPolicy::Backend::Tbb; ex.threads = th; ex.deterministic = true; ex.telemetry = false;
        const std::string label = std::string("tbb_threads=") + std::to_string(th) +
            ", verts=" + std::to_string(n * n) +
            ", edges~=" + std::to_string(edges.size()) +
            ", steps=" + std::to_string(steps_per_measure);

        BENCHMARK_ADVANCED(label.c_str())
        (Catch::Benchmark::Chronometer meter) {
            Handle h = create(InitDesc{std::span<const float>(xyz.data(), xyz.size()),
                                       std::span<const u32>(tris.data(), tris.size()),
                                       std::span<const u32>(fixed.data(), fixed.size()),
                                       ex, solve});
            REQUIRE(h != nullptr);
            for (int w = 0; w < 8; ++w) step(h, sp);
            meter.measure([&] {
                for (int i = 0; i < steps_per_measure; ++i) step(h, sp);
            });
            destroy(h);
        };
    }
}

