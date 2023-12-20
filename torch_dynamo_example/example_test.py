from typing import List

import torch
from torch import _dynamo as torchdynamo


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler called with fx_graph:\n", gm.graph.print_tabular())
    return gm.forward


@torchdynamo.optimize(my_compiler)
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        x = x * -1
    return x * b


def test():
    toy_example(torch.randn(3), torch.randn(3))


if __name__ == "__main__":
    test()
