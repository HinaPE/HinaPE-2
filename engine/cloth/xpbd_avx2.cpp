#include "cloth.h"

namespace HinaPE {
    namespace detail {

#if defined(HINAPE_HAVE_AVX2)
        ISim* make_avx2(const InitDesc& desc) {
            // Currently fallback to native; hook for AVX2 specialization is preserved.
            return make_native(desc);
        }
#endif

    } // namespace detail
} // namespace HinaPE
