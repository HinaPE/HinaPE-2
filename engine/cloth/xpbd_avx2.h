#ifndef HINAPE_XPBD_AVX2_H
#define HINAPE_XPBD_AVX2_H

#include "cloth.h"

namespace HinaPE::detail {
#if defined(HINAPE_HAVE_AVX2)
    ISim* make_avx2(const InitDesc& desc);
#endif
}

#endif // HINAPE_XPBD_AVX2_H

