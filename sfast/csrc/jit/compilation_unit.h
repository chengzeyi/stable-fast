#pragma once
#include <torch/extension.h>

namespace sfast {
namespace jit {

using namespace torch;
using namespace torch::jit;

void ClearClassTypeRegistration(const ClassTypePtr &cls_type);
void ClearModuleRegistration(const Module &module);

} // namespace jit
} // namespace sfast
