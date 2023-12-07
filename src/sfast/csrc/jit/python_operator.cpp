#include <torch/extension.h>

#include <ATen/core/dispatch/OperatorOptions.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/resource_guard.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/utils/tensor_memoryformats.h>
#include <torch/library.h>

#include "python_operator.h"

namespace sfast {
namespace jit {

using namespace torch::jit;

void RegisterCustomPythonOperator(const std::string &schema,
                                  THPObjectPtr &&py_callable) {
  FunctionSchema parsed_schema = parseSchema(schema);
  auto name = parsed_schema.name();
  auto arguments = parsed_schema.arguments();
  auto returns = parsed_schema.returns();

  std::shared_ptr<py::function> func_ptr(
      new py::function(py::reinterpret_borrow<const py::function>(
          py::handle(const_cast<PyObject *>(py_callable.get())))),
      [](py::function *ptr) {
        // Check if the current thread is holding the GIL
        if (PyGILState_Check()) {
          delete ptr;
        } else {
          py::gil_scoped_acquire gil;
          delete ptr;
        }
      });

  RegisterOperators({Operator(
      schema,
      [=](Stack &stack) {
        pybind11::gil_scoped_acquire gil;

        bool tracing = torch::jit::tracer::isTracing();
        Node *node;
        if (tracing) {
          node = torch::jit::tracer::getTracingState()->createNode(
              c10::Symbol::fromQualString(name), returns.size());
          torch::jit::tracer::recordSourceLocation(node);
        }

        size_t num_inputs = arguments.size();
        size_t num_outputs = returns.size();

        auto inputs = pop(stack, num_inputs);
        py::tuple py_inputs(num_inputs);
        for (size_t i = 0; i < num_inputs; ++i) {
          if (tracing) {
            node->addInput(torch::jit::tracer::getValueTrace(inputs[i]));
          }
          if (inputs[i].isInt() && arguments[i].name() == "memory_format" &&
              arguments[i].type()->kind() == c10::TypeKind::IntType) {
            // auto memory_format = torch::utils::getTHPMemoryFormat(
            //     static_cast<c10::MemoryFormat>(inputs[i].toInt()));
            // py_inputs[i] = py::handle(memory_format);
            py_inputs[i] =
                py::cast(static_cast<c10::MemoryFormat>(inputs[i].toInt()));
          } else {
            py_inputs[i] = toPyObject(inputs[i]);
          }
        }
        try {
          py::object output;
          {
            ResourceGuard guard(torch::jit::tracer::pauseTracing());
            output = (*func_ptr)(*py_inputs);
          }
          if (num_outputs == 1) {
            auto output_type = returns[0].type();
            push(stack, returnToIValue(output_type, output));
            if (tracing) {
              auto output_val = node->output();
              torch::jit::tracer::setValueTrace(stack.back(), output_val);
            }
          } else {
            auto outputs = output.cast<py::tuple>();
            for (size_t i = 0; i < num_outputs; ++i) {
              push(stack, returnToIValue(returns[i].type(), outputs[i]));
              if (tracing) {
                auto output_val = node->outputs()[i];
                torch::jit::tracer::setValueTrace(stack.back(), output_val);
              }
            }
          }
        } catch (py::error_already_set &e) {
          throw std::runtime_error(e.what());
        }

        if (tracing) {
          torch::jit::tracer::getTracingState()->insertNode(node);
        }
      },
      c10::AliasAnalysisKind::FROM_SCHEMA)});
}

} // namespace jit
} // namespace sfast
