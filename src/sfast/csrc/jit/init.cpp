#include <torch/extension.h>

#include <ATen/core/dispatch/OperatorOptions.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <torch/csrc/MemoryFormat.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/custom_operator.h>

#include "compilation_unit.h"
#include "device_constant_override.h"
#include "frozen_conv_folding_fix.h"
#include "op_input_tensor_conversion.h"
#include "python_operator.h"
#include "scalar_tensor_erase.h"
#include "simple_arith_elimination.h"

namespace sfast {
namespace jit {

using namespace torch::jit;

void initJITBindings(py::module m) {
  m.def("_jit_pass_eliminate_simple_arith", EliminateSimpleArith);
  m.def("_jit_pass_erase_scalar_tensors", EraseScalarTensors);
  m.def("_jit_pass_fix_frozen_conv_folding", FixFrozenConvFolding);
  m.def("_jit_pass_override_device_constants", OverrideDeviceConstants);
  m.def(
      "_jit_pass_convert_op_input_tensors",
      [](std::shared_ptr<torch::jit::Graph> &graph, const std::string &op_name,
         const c10::optional<at::Device> &device,
         const c10::optional<py::object> &data_type_obj,
         const c10::optional<py::object> &memory_format_obj,
         const c10::optional<std::vector<int>> &indices) {
        c10::optional<at::ScalarType> dtype;
        if (data_type_obj.has_value()) {
          dtype.emplace(
              reinterpret_cast<THPDtype *>(data_type_obj.value().ptr())
                  ->scalar_type);
        }
        c10::optional<at::MemoryFormat> memory_format;
        if (memory_format_obj.has_value()) {
          memory_format.emplace(reinterpret_cast<THPMemoryFormat *>(
                                    memory_format_obj.value().ptr())
                                    ->memory_format);
        }
        ConvertOpInputTensors(graph, op_name, device, dtype, memory_format,
                              indices);
      },
      py::arg("graph"), py::arg("op_name"), py::arg("device") = py::none(),
      py::arg("dtype") = py::none(), py::arg("memory_format") = py::none(),
      py::arg("indices") = py::none());
  m.def("_jit_replace_graph",
        [](std::shared_ptr<torch::jit::Graph> &graph,
           const std::shared_ptr<torch::jit::Graph> &replacement) {
          GRAPH_DUMP("Original Graph", graph);
          graph->block()->clear();
          graph->block()->cloneFrom(replacement->block(), nullptr);
          GRAPH_DUMP("Replaced Graph", graph);
        });
  m.def("_jit_clear_graph", [](std::shared_ptr<torch::jit::Graph> &graph) {
    GRAPH_DUMP("Original Graph", graph);
    graph->block()->clear();
  });
  m.def("_jit_clear_class_type_registration", ClearClassTypeRegistration);
  m.def("_jit_clear_module_registration", ClearModuleRegistration);
  m.def("_jit_get_module_type",
        [](const Module &module) { return module.type(); });
  m.def("_jit_register_custom_python_operator",
        [](const std::string &schema, py::object &py_callable) {
          THPObjectPtr py_callable_ptr(py_callable.ptr());
          RegisterCustomPythonOperator(schema, std::move(py_callable_ptr));
        });
}

} // namespace jit
} // namespace sfast
