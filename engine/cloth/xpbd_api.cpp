#include "cloth.h"
namespace HinaPE {
    Handle create(const InitDesc& desc) {
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
