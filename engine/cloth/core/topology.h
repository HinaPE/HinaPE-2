#ifndef HINAPE_CLOTH_CORE_TOPOLOGY_H
#define HINAPE_CLOTH_CORE_TOPOLOGY_H

#include "cloth.h"
#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

namespace HinaPE::topo {

    struct BitsetDyn {
        std::vector<std::uint64_t> w;
        void ensure(int bit) {
            usize need = static_cast<usize>(bit / 64 + 1);
            if (w.size() < need) w.resize(need, 0);
        }
        void set(int bit) {
            ensure(bit);
            w[static_cast<usize>(bit / 64)] |= (std::uint64_t(1) << (bit & 63));
        }
        [[nodiscard]] bool test(int bit) const noexcept {
            usize idx = static_cast<usize>(bit / 64);
            if (idx >= w.size()) return false;
            return (w[idx] >> (bit & 63)) & 1ULL;
        }
    };

    inline void build_edges_from_triangles(std::span<const u32> tris, std::vector<std::pair<u32, u32>>& edges) {
        const usize m = tris.size() / 3;
        edges.clear();
        edges.reserve(m * 3);
        for (usize t = 0; t < m; ++t) {
            u32 a = tris[t * 3 + 0], b = tris[t * 3 + 1], c = tris[t * 3 + 2];
            auto add = [&](u32 i, u32 j) {
                if (i > j) std::swap(i, j);
                edges.emplace_back(i, j);
            };
            add(a, b);
            add(b, c);
            add(c, a);
        }
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    }

    inline void build_adjacency(usize n, const std::vector<std::pair<u32, u32>>& edges, std::vector<std::vector<u32>>& adj) {
        adj.assign(n, {});
        for (u32 e = 0; e < static_cast<u32>(edges.size()); ++e) {
            auto [i, j] = edges[e];
            adj[i].push_back(e);
            adj[j].push_back(e);
        }
    }

    inline void greedy_edge_coloring(usize n, const std::vector<std::pair<u32, u32>>& edges, const std::vector<std::vector<u32>>& adj, std::vector<std::vector<u32>>& batches) {
        const usize m = edges.size();
        std::vector<u32> order(m);
        std::iota(order.begin(), order.end(), 0);
        std::vector<usize> deg(n);
        for (usize v = 0; v < n; ++v) deg[v] = adj[v].size();
        std::sort(order.begin(), order.end(), [&](u32 a, u32 b) {
            auto [ai, aj] = edges[a];
            auto [bi, bj] = edges[b];
            usize da      = std::max(deg[ai], deg[aj]);
            usize db      = std::max(deg[bi], deg[bj]);
            if (da != db) return da > db;
            return a < b;
        });
        std::vector<BitsetDyn> used(n);
        int maxc = -1;
        std::vector<int> color(m, -1);
        for (u32 idx : order) {
            auto [i, j] = edges[idx];
            int c       = 0;
            for (;;) {
                if (!used[i].test(c) && !used[j].test(c)) break;
                ++c;
            }
            used[i].set(c);
            used[j].set(c);
            color[idx] = c;
            if (c > maxc) maxc = c;
        }
        const usize nb = static_cast<usize>(maxc + 1);
        batches.assign(nb, {});
        for (u32 e = 0; e < static_cast<u32>(m); ++e) batches[static_cast<usize>(color[e])].push_back(e);
    }

}

#endif
