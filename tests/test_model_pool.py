import typing

import torch

from torch_dynamo_example.model_pool import huggingface_model


def run_model(model: torch.nn.Module, example_input: typing.Dict[str, torch.Tensor]):
    model(**example_input)


def test_huggingface_model():
    for name, model_fn in huggingface_model.model_list.items():
        model, example_input = model_fn()
        run_model(model, example_input)
