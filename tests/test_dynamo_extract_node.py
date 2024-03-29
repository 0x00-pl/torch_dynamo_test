import typing

import torch
from torch import _dynamo as torchdynamo
from torch.fx.experimental.proxy_tensor import make_fx

from torch_dynamo_test.graph_visiter.modify_node_visitor import ModifyNodeTorchVisitor
from torch_dynamo_test.model_pool import huggingface_model


class ModifyAddNodeToMyVersionVisitor(ModifyNodeTorchVisitor):
    def before_node(self, node: torch.fx.Node):
        print(
            f'before node {node}, target {node.target}  comapre to {torch.ops.aten.add.Tensor}  result {node.target is torch.ops.aten.add.Tensor}'
        )
        if node.target is torch.ops.aten.add.Tensor:
            print('found torch.add')
            node.target = self.my_add
        return node

    @staticmethod
    @torchdynamo.allow_in_graph
    def my_add(a, b):
        # res = torch.tensor([float(ai) + float(bi) for ai, bi in zip(a, b)])
        print('running my add.')
        res = a + b
        return res


class ModifyMulNodeToMyVersionVisitor(ModifyNodeTorchVisitor):
    def before_node(self, node: torch.fx.Node):
        if node.target is torch.ops.aten.mul.Tensor:
            print('found torch.mul')
            node.target = self.my_mul
        elif node.target is torch.ops.aten.bmm.default:
            print('found torch.bmm')
            node.target = self.my_bmm
        return node

    @staticmethod
    @torchdynamo.allow_in_graph
    def my_mul(a, b):
        # res = torch.tensor([float(ai) + float(bi) for ai, bi in zip(a, b)])
        print('running my mul.')
        res = a * b
        return res

    @staticmethod
    @torchdynamo.allow_in_graph
    def my_bmm(a, b):
        # res = torch.tensor([float(ai) + float(bi) for ai, bi in zip(a, b)])
        print('running my bmm.')
        res = torch.bmm(a, b)
        return res


def my_compiler(gm: torch.fx.GraphModule, example_inputs: typing.List[torch.Tensor]):
    gm_fx = make_fx(gm, tracing_mode="real")(*example_inputs)

    print()
    gm_fx.print_readable()

    new_gm_fx = ModifyMulNodeToMyVersionVisitor().visit_module(gm_fx)

    print()
    new_gm_fx.print_readable()

    return new_gm_fx.forward


# torchdynamo.disallow_in_graph(torch.div)

@torchdynamo.optimize(my_compiler)
def toy_example(a, b):
    x = torch.div(a, (torch.abs(a) + 1))
    return (x * b) - 1


def test_dynamo_extract_node():
    torchdynamo.reset()
    for _ in range(10):
        toy_example(torch.randn(3), torch.randn(3))


@torchdynamo.optimize(my_compiler)
def run_model(model: torch.nn.Module, example_input: typing.Dict[str, torch.Tensor]):
    model(**example_input)


def test_huggingface_model():
    for name, model_fn in huggingface_model.model_fn_list.items():
        model, example_input = model_fn()
        run_model(model, example_input)
