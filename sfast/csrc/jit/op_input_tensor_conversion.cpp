#include <torch/extension.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inliner.h>

#include "op_input_tensor_conversion.h"

namespace sfast {
namespace jit {

using namespace torch::jit;

void ConvertOpInputTensors(std::shared_ptr<Graph> &graph,
                           const std::string &op_name,
                           const c10::optional<at::Device> &device,
                           const c10::optional<at::ScalarType> &dtype,
                           const c10::optional<at::MemoryFormat> &memory_format,
                           const c10::optional<std::vector<int>> &indices) {
  ConvertOpInputTensorsOnBlock(graph->block(), op_name, device, dtype,
                               memory_format, indices);
  GRAPH_DUMP("After ConvertOpInputTensors: ", graph);
}

void ConvertOpInputTensorsOnBlock(
    Block *block, const std::string &op_name,
    const c10::optional<at::Device> &device,
    const c10::optional<at::ScalarType> &dtype,
    const c10::optional<at::MemoryFormat> &memory_format,
    const c10::optional<std::vector<int>> &indices) {
  static torch::jit::CompilationUnit decompose_funcs(R"SCRIPT(
def tensor_to(self: Tensor, device: Optional[Device], dtype: Optional[int], memory_format: Optional[int]):
    return torch.to(self, dtype=dtype, layout=None, device=device, memory_format=memory_format)

def list_of_tensors_to(self: List[Tensor], device: Optional[Device], dtype: Optional[int], memory_format: Optional[int]):
    l: List[Tensor] = []
    for t in self:
        l.append(torch.to(t, dtype=dtype, layout=None, device=device, memory_format=memory_format))
    return l

def list_of_optional_tensors_to(self: List[Optional[Tensor]], device: Optional[Device], dtype: Optional[int], memory_format: Optional[int]):
    l: List[Optional[Tensor]] = []
    for t in self:
        if t is None:
            l.append(t)
        else:
            l.append(torch.to(t, dtype=dtype, layout=None, device=device, memory_format=memory_format))
    return l
)SCRIPT");

  for (auto it = block->nodes().begin(), end = block->nodes().end(); it != end;
       ++it) {
    for (auto sub : it->blocks()) {
      ConvertOpInputTensorsOnBlock(sub, op_name, device, dtype, memory_format,
                                   indices);
    }

    if (it->kind().toQualString() == op_name) {
      size_t idx = 0;
      for (auto input : it->inputs()) {
        if ((!indices.has_value() ||
             std::find(indices.value().begin(), indices.value().end(), idx) !=
                 indices.value().end()) &&
            (input->type()->isSubtypeOf(TensorType::get()) ||
             input->type()->isSubtypeOf(ListType::ofTensors()) ||
             input->type()->isSubtypeOf(ListType::ofOptionalTensors()))) {
          WithInsertPoint guard(*it);

          std::string func_name = "tensor_to";
          if (input->type()->isSubtypeOf(ListType::ofTensors())) {
            func_name = "list_of_tensors_to";
          } else if (input->type()->isSubtypeOf(
                         ListType::ofOptionalTensors())) {
            func_name = "list_of_optional_tensors_to";
          }

          std::shared_ptr<Graph> d_graph =
              toGraphFunction(decompose_funcs.get_function(func_name)).graph();
          Inline(*d_graph);
          EliminateDeadCode(d_graph);
          c10::optional<int64_t> casted_memory_format;
          if (memory_format.has_value()) {
            casted_memory_format.emplace(
                static_cast<std::underlying_type<at::MemoryFormat>::type>(
                    memory_format.value()));
          }
          std::vector<Value *> inputs{
              input,
              block->owningGraph()->insertConstant(device, c10::nullopt,
                                                   it->scope()),
              block->owningGraph()->insertConstant(dtype, c10::nullopt,
                                                   it->scope()),
              block->owningGraph()->insertConstant(casted_memory_format,
                                                   c10::nullopt, it->scope())};
          Value *r = insertGraph(*block->owningGraph(), *d_graph, inputs).at(0);
          r->setType(input->type());
          it->replaceInput(idx, r);
        }
        ++idx;
      }
    }
  }
}

} // namespace jit
} // namespace sfast
