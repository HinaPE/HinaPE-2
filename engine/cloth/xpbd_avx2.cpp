#include "cloth.h"
namespace HinaPE { namespace detail {
ISim* make_avx2(const InitDesc& desc) { return make_native(desc); }
} }
