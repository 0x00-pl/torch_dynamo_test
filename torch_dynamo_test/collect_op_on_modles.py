import typing

import torch
from torch import _dynamo
from torch.fx.experimental.proxy_tensor import make_fx

from torch_dynamo_example.graph_visiter import collect_ops
from torch_dynamo_example.model_pool import huggingface_model


def collect_op():
    collector = collect_ops.CollectOps()

    def my_compiler(gm: torch.fx.GraphModule, example_inputs: typing.List[torch.Tensor]):
        gm = make_fx(gm, tracing_mode="real")(*example_inputs)
        collector.visit_module(gm)
        return gm.forward

    for name, model_fn in huggingface_model.model_fn_list.items():
        model, example_input = model_fn()
        model = _dynamo.optimize(my_compiler)(model)
        model(**example_input)

    print('opset: ', collector.call_function_names, ' ', collector.call_module_names, ' ', collector.call_method_names)
