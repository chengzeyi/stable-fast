#pragma once
#include <torch/extension.h>

namespace sfast {
namespace jit {

void initJITBindings(py::module m);

} // namespace jit
} // namespace sfast
