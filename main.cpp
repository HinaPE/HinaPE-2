#include "engine/cloth/xpbd.h"
#include <vector>
#include <cstdio>

using namespace HinaPE;

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
    for (int x = 0; x < nx; ++x) fixed.push_back(static_cast<u32>(x));
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

int main() {
    std::vector<float> xyz;
    std::vector<u32> tris;
    std::vector<u32> fixed;
    make_grid(32, 32, 0.05f, xyz, tris, fixed);
    ExecPolicy exec{};
    SolvePolicy solve{};
    solve.substeps = 2;
    solve.iterations = 12;
    solve.compliance_stretch = 0.0f;
    solve.damping = 0.02f;
    InitDesc init{std::span<const float>(xyz.data(), xyz.size()), std::span<const u32>(tris.data(), tris.size()), std::span<const u32>(fixed.data(), fixed.size()), exec, solve};
    Handle h = create(init);
    StepParams sp{};
    sp.dt = 1.0f/60.0f;
    sp.gravity_y = -9.81f;
    for (int i = 0; i < 240; ++i) step(h, sp);
    auto v = map_dynamic(h);
    usize mid = static_cast<usize>((32/2) * 32 + (32/2));
    std::printf("pos %f %f %f", v.pos_x[mid], v.pos_y[mid], v.pos_z[mid]);
    destroy(h);
    return 0;
}
