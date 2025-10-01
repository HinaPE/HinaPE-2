#ifndef HINAPE_CLOTH_CORE_ARENA_H
#define HINAPE_CLOTH_CORE_ARENA_H

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory_resource>
#include <new>
#include <vector>

namespace HinaPE::core {

    class aligned_resource final : public std::pmr::memory_resource {
    public:
        explicit aligned_resource(std::size_t alignment = 64, std::pmr::memory_resource* upstream = std::pmr::get_default_resource()) : align_(normalize_alignment(alignment)), upstream_(upstream) {}

    private:
        std::size_t align_{};
        std::pmr::memory_resource* upstream_{};

        static std::size_t normalize_alignment(std::size_t a) {
            constexpr std::size_t base = alignof(void*);
            if (a < base) {
                a = base;
            }
            if ((a & (a - 1)) != 0) {
                std::size_t p = 1;
                while (p < a) {
                    p <<= 1U;
                }
                a = p;
            }
            return a;
        }

        void* do_allocate(std::size_t bytes, std::size_t alignment) override {
            const std::size_t req = normalize_alignment(std::max(align_, alignment));
            if (req <= alignof(std::max_align_t)) {
                return upstream_->allocate(bytes == 0 ? sizeof(std::max_align_t) : bytes, req);
            }
#if defined(_MSC_VER)
            void* p = _aligned_malloc(bytes == 0 ? req : bytes, req);
            if (!p) {
                throw std::bad_alloc{};
            }
            return p;
#else
            void* p       = nullptr;
            const auto sz = static_cast<std::size_t>(bytes == 0 ? req : bytes);
            if (posix_memalign(&p, req, sz) != 0) {
                throw std::bad_alloc{};
            }
            return p;
#endif
        }

        void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
            if (!p) {
                return;
            }
            const std::size_t req = normalize_alignment(std::max(align_, alignment));
            if (req <= alignof(std::max_align_t)) {
                upstream_->deallocate(p, bytes == 0 ? sizeof(std::max_align_t) : bytes, req);
                return;
            }
#if defined(_MSC_VER)
            _aligned_free(p);
#else
            std::free(p);
#endif
        }

        [[nodiscard]] bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
            return this == &other;
        }
    };

    template <class T>
    using pvec = std::pmr::vector<T>;

}

#endif
