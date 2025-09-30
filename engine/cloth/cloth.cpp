#include "cloth.h"
namespace HinaPE {
Handle create(const InitDesc& desc) {
    using Backend = ExecPolicy::Backend;
    if (desc.exec.backend == Backend::Avx2) {
#if defined(HINAPE_HAVE_AVX2)
        return detail::make_avx2(desc);
#else
        return detail::make_native(desc);
#endif
    }
    if (desc.exec.backend == Backend::Tbb) {
#if defined(HINAPE_HAVE_TBB)
        return detail::make_tbb(desc);
#else
        return detail::make_native(desc);
#endif
    }
    return detail::make_native(desc);
}
void destroy(Handle h) noexcept { delete h; }
void step(Handle h, const StepParams& params) noexcept { if (h) h->step(params); }
DynamicView map_dynamic(Handle h) noexcept { if (!h) return {}; return h->map_dynamic(); }
}
