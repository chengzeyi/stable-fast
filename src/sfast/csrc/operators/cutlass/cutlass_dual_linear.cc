#include <torch/extension.h>

#include "cutlass_dual_linear.h"
#include "cutlass_dual_linear_kernel.h"

namespace sfast {
namespace operators {

void initCutlassDualLinearBindings(torch::Library &m) {
  m.def("cutlass_linear_geglu",
        torch::dispatch(c10::DispatchKey::CompositeImplicitAutograd,
                        cutlass_linear_geglu));
  m.def("cutlass_linear_geglu_unified",
        torch::dispatch(c10::DispatchKey::CompositeImplicitAutograd,
                        cutlass_linear_geglu_unified));
}

} // namespace operators
} // namespace sfast
