#include <torch/extension.h>

#include <ATen/core/dispatch/OperatorOptions.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/utils/tensor_memoryformats.h>
#include <torch/library.h>

#include "python_operator.h"

namespace sfast {
namespace jit {

using namespace torch::jit;
using graph_rewrite_helper::PatternInfo;

void ReplaceByPythonOperator(std::shared_ptr<Graph> &graph,
                             const std::string &op_name, THPObjectPtr &pyobj,
                             const std::string &arg_types,
                             pyobj_list &scalar_args) {
  ReplaceByPythonOperatorOnBlock(graph->block(), op_name, pyobj, arg_types,
                                 scalar_args);
  GRAPH_DUMP("After ReplacePythonOperator: ", graph);
}

void ReplaceByPythonOperatorOnBlock(Block *block, const std::string &op_name,
                                    THPObjectPtr &pyobj,
                                    const std::string &arg_types,
                                    pyobj_list &scalar_args) {
  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks()) {
      ReplaceByPythonOperatorOnBlock(sub, op_name, pyobj, arg_types,
                                     scalar_args);
    }

    if (it->kind().toQualString() == op_name) {
      THPObjectPtr apply(PyObject_GetAttrString(pyobj.get(), "apply"));
      if (!apply) {
        throw python_error();
      }
      WithInsertPoint guard(*it);
      pyobj_list scalar_args_copy;
      scalar_args_copy.reserve(scalar_args.size());
      for (auto &arg : scalar_args) {
        auto raw_ptr = arg.get();
        Py_INCREF(raw_ptr);
        scalar_args_copy.emplace_back(raw_ptr);
      }
      Node *python_node = block->owningGraph()->createPythonOp(
          std::move(apply), arg_types, std::move(scalar_args_copy));
      python_node->insertBefore(*it);
      for (auto input : it->inputs()) {
        python_node->addInput(input);
      }
      it->output()->replaceAllUsesWith(python_node->output());
      it.destroyCurrent();
    }
  }
}

void RegisterCustomPythonOperator(const std::string &schema,
                                  THPObjectPtr &&py_callable) {
  FunctionSchema parsed_schema = parseSchema(schema);
  auto arguments = parsed_schema.arguments();
  auto returns = parsed_schema.returns();

  const py::function func = py::reinterpret_borrow<const py::function>(
      py::handle(const_cast<PyObject *>(py_callable.get())));

  RegisterOperators({Operator(
      schema,
      [=](Stack &stack) {
        pybind11::gil_scoped_acquire gil;
        size_t num_inputs = arguments.size();
        size_t num_outputs = returns.size();

        auto inputs = pop(stack, num_inputs);
        py::tuple py_inputs(num_inputs);
        for (size_t i = 0; i < num_inputs; ++i) {
          if (inputs[i].isInt() && arguments[i].name() == "memory_format" &&
              arguments[i].type()->kind() == c10::TypeKind::IntType) {
            auto memory_format = torch::utils::getTHPMemoryFormat(
                static_cast<c10::MemoryFormat>(inputs[i].toInt()));
            py_inputs[i] = py::handle(memory_format);
          } else {
            py_inputs[i] = toPyObject(inputs[i]);
          }
        }
        try {
          py::object output = func(*py_inputs);
          if (num_outputs == 1) {
            auto output_type = returns[0].type();
            push(stack, returnToIValue(output_type, output));
          } else {
            auto outputs = output.cast<py::tuple>();
            for (size_t i = 0; i < num_outputs; ++i) {
              push(stack, returnToIValue(returns[i].type(), outputs[i]));
            }
          }
        } catch (py::error_already_set &e) {
          throw std::runtime_error(e.what());
        }
      },
      c10::AliasAnalysisKind::FROM_SCHEMA)});
}

} // namespace jit
} // namespace sfast
