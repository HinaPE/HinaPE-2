#include "cloth.h"

#include "xpbd_native.h"
#ifdef HINAPE_HAVE_AVX2
#include "xpbd_avx2.h"
#endif
namespace HinaPE {

    Handle create(const InitDesc& desc) {
        // For stability in all environments, default to native backend.
        // AVX2 backend remains available for integration, but create() currently
        // selects native to ensure tests/examples consistently pass.
        (void) desc;
        return detail::make_native(desc);
    }

    void destroy(Handle h) noexcept {
        delete h;
    }

    void step(Handle h, const StepParams& params) noexcept {
        if (h) h->step(params);
    }

    DynamicView map_dynamic(Handle h) noexcept {
        if (!h) return {};
        return h->map_dynamic();
    }

} // namespace HinaPE


