#ifndef HINAPE_XPBD_NATIVE_H
#define HINAPE_XPBD_NATIVE_H

#include "cloth.h"

namespace HinaPE::detail {
    ISim* make_native(const InitDesc& desc);
}

#endif // HINAPE_XPBD_NATIVE_H

