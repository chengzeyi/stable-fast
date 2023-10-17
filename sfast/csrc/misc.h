#pragma once
#include <torch/extension.h>

namespace sfast {
namespace misc {

void initMiscBindings(py::module m);

} // namespace jit
} // namespace sfast
