import typing

import torch
from torch import _dynamo as torchdynamo
from torch.fx.experimental.proxy_tensor import make_fx


def my_compiler(gm: torch.fx.GraphModule, example_inputs: typing.List[torch.Tensor]):
    return gm.forward


def my_compiler_fx(gm: torch.fx.GraphModule, example_inputs: typing.List[torch.Tensor]):
    gm_fx = make_fx(gm, tracing_mode="real")(*example_inputs)
    return gm_fx.forward


@torchdynamo.optimize(my_compiler)
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


@torchdynamo.optimize(my_compiler_fx)
def toy_example_fx(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


def test_dynamo():
    torchdynamo.reset()
    for _ in range(10):
        toy_example(torch.randn(3), torch.randn(3))


def test_dynamo_fx():
    torchdynamo.reset()
    for _ in range(10):
        toy_example_fx(torch.randn(3), torch.randn(3))
