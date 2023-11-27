#pragma once
#include <torch/extension.h>

#include <torch/library.h>

namespace sfast {
namespace operators {

using namespace torch;

void initCutlassQLinearBindings(torch::Library &m)
#if defined(WITH_CUDA)
;
#else
{}
#endif

} // namespace operators
} // namespace sfast
