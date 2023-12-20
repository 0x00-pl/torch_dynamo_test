from typing import List

import torch
from torch import _dynamo as torchdynamo


def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    # print("my_compiler called with fx_graph:\n", gm.graph.print_tabular())
    print('=================')
    print(gm.code)
    return gm.forward


@torchdynamo.optimize(my_compiler)
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b


def test():
    result = toy_example(torch.randn(3), torch.randn(3))
    print('result is:', result)


if __name__ == "__main__":
    for _ in range(10):
        test()
