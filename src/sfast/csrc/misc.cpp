#include <torch/extension.h>

#include <ATen/TensorUtils.h>

#include "misc.h"

namespace sfast {
namespace misc {

using namespace torch;

void initMiscBindings(py::module m) {
  m.def("_abort", []() { std::abort(); });
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
  m.def("_create_shadow_tensor", [](const Tensor &tensor, bool detach) {
    auto output = torch::from_blob(tensor.data_ptr(), tensor.sizes(),
                                   tensor.strides(), tensor.options());
    output.requires_grad_(tensor.requires_grad());
    return output;
  }, py::arg("tensor"), py::arg("detach") = false);
}

} // namespace misc
} // namespace sfast
