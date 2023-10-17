#include <torch/extension.h>

#include <ATen/TensorUtils.h>

#include "misc.h"

namespace sfast {
namespace misc {

using namespace torch;

void initMiscBindings(py::module m) {
  m.def("_compute_stride", [](const IntArrayRef &oldshape,
                              const IntArrayRef &oldstride,
                              const IntArrayRef &newshape) {
    auto stride = at::detail::computeStride(oldshape, oldstride, newshape);

    c10::optional<IntArrayRef> stride_opt;
    if (stride.has_value()) {
      stride_opt = stride.value();
    }
    return stride_opt;
  });
}

} // namespace misc
} // namespace sfast
