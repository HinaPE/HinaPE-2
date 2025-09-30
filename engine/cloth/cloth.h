#ifndef HINAPE_CLOTH_H
#define HINAPE_CLOTH_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <algorithm>
#include <memory_resource>
#include <new>
#include <vector>
#include <numeric>

namespace HinaPE {

    using u32   = std::uint32_t;
    using f32   = float;
    using usize = std::size_t;

    struct ExecPolicy {
        enum class Backend { Native, Avx2 };
        Backend backend{Backend::Native};
        int threads{0};
        bool deterministic{true};
        bool telemetry{false};
    };

    struct SolvePolicy {
        int substeps{1};
        int iterations{8};
        f32 compliance_stretch{0.0f};
        f32 damping{0.01f};
    };

    struct InitDesc {
        std::span<const f32> positions_xyz;
        std::span<const u32> triangles;
        std::span<const u32> fixed_indices;
        ExecPolicy exec{};
        SolvePolicy solve{};
    };

    struct StepParams {
        f32 dt{1.0f / 60.0f};
        f32 gravity_x{0.0f};
        f32 gravity_y{-9.81f};
        f32 gravity_z{0.0f};
    };

    struct DynamicView {
        f32* pos_x{};
        f32* pos_y{};
        f32* pos_z{};
        f32* vel_x{};
        f32* vel_y{};
        f32* vel_z{};
        usize count{};
    };

    namespace detail {
        struct ISim {
            virtual ~ISim() = default;
            virtual void step(const StepParams&) noexcept = 0;
            virtual DynamicView map_dynamic() noexcept    = 0;
        };

        // Core memory resource (64B-aligned) and pmr alias
        class aligned_resource final : public std::pmr::memory_resource {
        public:
            explicit aligned_resource(std::size_t alignment = 64, std::pmr::memory_resource* upstream = std::pmr::get_default_resource())
                : align_(normalize_alignment(alignment)), upstream_(upstream) {}
        private:
            std::size_t align_{};
            std::pmr::memory_resource* upstream_{};
            static std::size_t normalize_alignment(std::size_t a) {
                constexpr std::size_t base = alignof(void*);
                if (a < base) a = base;
                if ((a & (a - 1)) != 0) {
                    std::size_t p = 1; while (p < a) p <<= 1U; a = p;
                }
                return a;
            }
            void* do_allocate(std::size_t bytes, std::size_t alignment) override {
                const std::size_t req = normalize_alignment(std::max(align_, alignment));
                if (req <= alignof(std::max_align_t)) return upstream_->allocate(bytes == 0 ? sizeof(std::max_align_t) : bytes, req);
        #if defined(_MSC_VER)
                void* p = _aligned_malloc(bytes == 0 ? req : bytes, req); if (!p) throw std::bad_alloc{}; return p;
        #else
                void* p = nullptr; const auto sz = static_cast<std::size_t>(bytes == 0 ? req : bytes); if (posix_memalign(&p, req, sz) != 0) throw std::bad_alloc{}; return p;
        #endif
            }
            void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
                if (!p) return; const std::size_t req = normalize_alignment(std::max(align_, alignment));
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

        // Factory functions implemented by each backend
        ISim* make_native(const InitDesc& desc);
    #if defined(HINAPE_HAVE_AVX2)
        ISim* make_avx2(const InitDesc& desc);
    #endif
    } // namespace detail

    // Public API over opaque handle
    using Handle = detail::ISim*;
    [[nodiscard]] Handle create(const InitDesc& desc);
    void destroy(Handle h) noexcept;
    void step(Handle h, const StepParams& params) noexcept;
    [[nodiscard]] DynamicView map_dynamic(Handle h) noexcept;

    namespace topo {
        struct BitsetDyn {
            std::vector<std::uint64_t> w;
            void ensure(int bit) {
                usize need = static_cast<usize>(bit / 64 + 1);
                if (w.size() < need) w.resize(need, 0);
            }
            void set(int bit) { ensure(bit); w[static_cast<usize>(bit / 64)] |= (std::uint64_t(1) << (bit & 63)); }
            [[nodiscard]] bool test(int bit) const noexcept {
                usize idx = static_cast<usize>(bit / 64);
                if (idx >= w.size()) return false; return (w[idx] >> (bit & 63)) & 1ULL;
            }
        };
        inline void build_edges_from_triangles(std::span<const u32> tris, std::vector<std::pair<u32, u32>>& edges) {
            const usize m = tris.size() / 3; edges.clear(); edges.reserve(m * 3);
            for (usize t = 0; t < m; ++t) {
                u32 a = tris[t * 3 + 0], b = tris[t * 3 + 1], c = tris[t * 3 + 2];
                u32 e0a = a < b ? a : b, e0b = a < b ? b : a;
                u32 e1a = b < c ? b : c, e1b = b < c ? c : b;
                u32 e2a = c < a ? c : a, e2b = c < a ? a : c;
                edges.emplace_back(e0a, e0b); edges.emplace_back(e1a, e1b); edges.emplace_back(e2a, e2b);
            }
            std::sort(edges.begin(), edges.end()); edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        }
        inline void build_adjacency(usize n, const std::vector<std::pair<u32, u32>>& edges, std::vector<std::vector<u32>>& adj) {
            adj.assign(n, {});
            for (u32 e = 0; e < static_cast<u32>(edges.size()); ++e) { auto [i, j] = edges[e]; adj[i].push_back(e); adj[j].push_back(e); }
        }
        inline void greedy_edge_coloring(usize n, const std::vector<std::pair<u32, u32>>& edges, const std::vector<std::vector<u32>>& adj, std::vector<std::vector<u32>>& batches) {
            const usize m = edges.size(); std::vector<u32> order(m); std::iota(order.begin(), order.end(), 0);
            std::vector<usize> deg(n); for (usize v = 0; v < n; ++v) deg[v] = adj[v].size();
            std::sort(order.begin(), order.end(), [&](u32 a, u32 b) {
                auto [ai, aj] = edges[a]; auto [bi, bj] = edges[b];
                usize da = std::max(deg[ai], deg[aj]); usize db = std::max(deg[bi], deg[bj]);
                if (da != db) return da > db; return a < b;
            });
            std::vector<BitsetDyn> used(n); int maxc = -1; std::vector<int> color(m, -1);
            for (u32 idx : order) { auto [i, j] = edges[idx]; int c = 0; for (;;) { if (!used[i].test(c) && !used[j].test(c)) break; ++c; } used[i].set(c); used[j].set(c); color[idx] = c; if (c > maxc) maxc = c; }
            const usize nb = static_cast<usize>(maxc + 1); batches.assign(nb, {});
            for (u32 e = 0; e < static_cast<u32>(m); ++e) batches[static_cast<usize>(color[e])].push_back(e);
        }
    } // namespace topo

} // namespace HinaPE

#endif // HINAPE_CLOTH_H