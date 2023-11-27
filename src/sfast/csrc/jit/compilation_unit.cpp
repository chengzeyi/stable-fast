#include <torch/extension.h>
#include <ATen/core/custom_class.h>

#include "compilation_unit.h"

namespace sfast {
namespace jit {

using namespace torch;
using namespace torch::jit;

void ClearClassTypeRegistration(const ClassTypePtr &cls_type) {
    auto &&cu = cls_type->compilation_unit();

    for (auto &&method : cls_type->methods()) {
        auto &&qualname = method->qualname();
        if (cu->find_function(qualname) != nullptr) {
            cu->unsafeRemoveMethod(qualname);
        }
    }
}

void ClearModuleRegistration(const Module &module) {
    ClearClassTypeRegistration(module.type());

    for (auto &&submodule : module.named_children()) {
        ClearModuleRegistration(submodule.value);
    }
}

} // namespace jit
} // namespace sfast
