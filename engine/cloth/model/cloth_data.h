#ifndef HINAPE_CLOTH_MODEL_CLOTH_DATA_H
#define HINAPE_CLOTH_MODEL_CLOTH_DATA_H

#include "cloth.h"
#include "cloth/core/arena.h"
#include <memory_resource>

namespace HinaPE::model {

    using core::pvec;

    struct ParticleView {
        float* px{};
        float* py{};
        float* pz{};
        float* vx{};
        float* vy{};
        float* vz{};
        float* inv_mass{};
        size_t n{};
    };
    struct DistanceView {
        const u32* i{};
        const u32* j{};
        size_t m{};
        const float* rest{};
    };

    class ClothData {
    public:
        explicit ClothData(std::pmr::memory_resource* mr) : px(mr), py(mr), pz(mr), vx(mr), vy(mr), vz(mr), inv_mass(mr), e_i(mr), e_j(mr), rest_len(mr) {}

        void resize_particles(size_t n, bool pad8 = true) {
            n_logical    = n;
            size_t n_pad = pad8 ? ((n + 7u) & ~size_t(7)) : n;
            px.resize(n_pad);
            py.resize(n_pad);
            pz.resize(n_pad);
            vx.resize(n_pad);
            vy.resize(n_pad);
            vz.resize(n_pad);
            inv_mass.resize(n_pad);
            for (size_t i = n; i < n_pad; ++i) {
                px[i] = py[i] = pz[i] = 0.0f;
                vx[i] = vy[i] = vz[i] = 0.0f;
                inv_mass[i]           = 0.0f;
            }
        }
        void resize_edges(size_t m) {
            e_i.resize(m);
            e_j.resize(m);
            rest_len.resize(m);
        }

        [[nodiscard]] ParticleView particles() noexcept {
            return ParticleView{px.data(), py.data(), pz.data(), vx.data(), vy.data(), vz.data(), inv_mass.data(), n_logical};
        }
        [[nodiscard]] DistanceView distances() const noexcept {
            return DistanceView{e_i.data(), e_j.data(), e_i.size(), rest_len.data()};
        }

        pvec<float> px, py, pz, vx, vy, vz, inv_mass;
        pvec<u32> e_i, e_j;
        pvec<float> rest_len;
        size_t n_logical{0};
    };

}

#endif
