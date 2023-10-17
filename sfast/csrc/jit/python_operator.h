#pragma once
#include <torch/extension.h>

#include <torch/csrc/jit/ir/ir.h>

namespace sfast {
namespace jit {

using namespace torch::jit;

void ReplaceByPythonOperator(std::shared_ptr<Graph> &graph,
                             const std::string &op_name, THPObjectPtr &pyobj,
                             const std::string &arg_types,
                             pyobj_list &scalar_args);

void ReplaceByPythonOperatorOnBlock(Block *block, const std::string &op_name,
                                    THPObjectPtr &pyobj,
                                    const std::string &arg_types,
                                    pyobj_list &scalar_args);

void RegisterCustomPythonOperator(const std::string &schema,
                                  THPObjectPtr &&py_callable);

} // namespace jit
} // namespace sfast
