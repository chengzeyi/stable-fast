#pragma once
#include <torch/extension.h>

#include <torch/csrc/jit/ir/ir.h>

namespace sfast {
namespace jit {

using namespace torch::jit;

void RegisterCustomPythonOperator(const std::string &schema,
                                  THPObjectPtr &&py_callable);

} // namespace jit
} // namespace sfast
