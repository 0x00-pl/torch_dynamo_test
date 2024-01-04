import numpy as np
import torch

from torch_dynamo_test.graph_visiter.visitor import BaseTorchVisitor


class CollectOps(BaseTorchVisitor):
    def __init__(self, np_filename=None):
        super().__init__()
        self.call_function_names = set()
        self.call_module_names = set()
        self.call_method_names = set()
        self.np_filename = np_filename
        if np_filename is not None:
            self.import_file(np_filename)

    def before_node(self, node: torch.fx.Node):
        if node.op == 'call_function':
            self.call_function_names.add(node._pretty_print_target(node.target))
        elif node.op == 'call_module':
            self.call_module_names.add(node._pretty_print_target(node.target))
        elif node.op == 'call_method':
            self.call_method_names.add(node._pretty_print_target(node.target))
        else:
            pass

    def export_file(self, np_filename=None):
        if np_filename is None:
            np_filename = self.np_filename
        np.savez(
            np_filename,
            call_function_names=sorted(self.call_function_names),
            call_module_names=sorted(self.call_module_names),
            call_method_names=sorted(self.call_method_names)
        )

    def import_file(self, np_filename):
        data = np.load(np_filename)
        self.call_function_names = set(data['call_function_names'])
        self.call_module_names = set(data['call_module_names'])
        self.call_method_names = set(data['call_method_names'])
