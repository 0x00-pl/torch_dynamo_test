import typing

import torch

from torch_dynamo_test.graph_visiter import visitor


class ModifyNodeTorchVisitor(visitor.BaseTorchVisitor):
    def __init__(self):
        super().__init__()

    def visit_module(self, graph_module: torch.fx.GraphModule):
        new_module = self.before_module(graph_module)
        if new_module is None:
            new_graph = self.visit_graph(graph_module.graph)
            new_module = torch.fx.GraphModule(graph_module, new_graph)
        new_module = self.try_update_result(new_module, self.after_module(graph_module, new_module))

        return new_module

    def visit_graph(self, graph: torch.fx.Graph):
        new_graph = self.before_graph(graph)
        if new_graph is None:
            new_graph = torch.fx.Graph()
            with torch.fx.Node as Node:
                env: typing.Dict[Node, Node] = {}
            for node in graph.nodes:
                new_node = self.visit_node(node)
                env[node] = new_graph.node_copy(new_node, lambda n: env[n])

        new_graph = self.try_update_result(new_graph, self.after_graph(graph, new_graph))
        return new_graph

    def visit_node(self, node: torch.fx.Node):
        new_node = self.before_node(node)
        if new_node is None:
            new_node = node
        new_node = self.try_update_result(new_node, self.after_node(node, new_node))
        return new_node
