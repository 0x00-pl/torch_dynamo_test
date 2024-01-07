import torch


class BaseTorchVisitor:
    def __init__(self):
        pass

    @staticmethod
    def try_update_result(result, new_result):
        return result if new_result is None else new_result

    def visit_module(self, graph_module: torch.fx.GraphModule):
        result = self.before_module(graph_module)
        self.visit_graph(graph_module.graph)
        result = self.try_update_result(result, self.after_module(graph_module, result))
        return result

    def visit_graph(self, graph: torch.fx.Graph):
        result = self.before_graph(graph)
        for i in graph.nodes:
            self.visit_node(i)
        result = self.try_update_result(result, self.after_graph(graph, result))
        return result

    def visit_node(self, node: torch.fx.Node):
        result = self.before_node(node)
        for i in node.args:
            if isinstance(i, torch.fx.Node):
                self.visit_node(i)
            elif isinstance(i, tuple):
                for j in i:
                    if isinstance(j, torch.fx.Node):
                        self.visit_node(j)
        for k, v in node.kwargs.items():
            if isinstance(v, torch.fx.Node):
                self.visit_node(v)
        result = self.try_update_result(result, self.after_node(node, result))
        return result

    def before_module(self, graph_module: torch.fx.GraphModule):
        _, _ = self, graph_module
        return None

    def after_module(self, graph_module: torch.fx.GraphModule, result):
        _, _ = self, graph_module
        return result

    def before_graph(self, graph: torch.fx.Graph):
        _, _ = self, graph
        return None

    def after_graph(self, graph: torch.fx.Graph, result):
        _, _ = self, graph
        return result

    def before_node(self, node: torch.fx.Node):
        _, _ = self, node
        return None

    def after_node(self, node: torch.fx.Node, result):
        _, _ = self, node
        return result
