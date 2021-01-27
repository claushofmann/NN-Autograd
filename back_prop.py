from graph import Node, ValueNode
import numpy as np


class BackProp:
    def __init__(self, target_node):
        self.target_node = target_node

    def compute_gradients(self, diff_nodes, gradient_table=None):
        if gradient_table is None:
            gradient_table = dict()

        for diff_node in diff_nodes:
            if diff_node not in gradient_table:
                consumers = [consumer[1] for consumer in diff_node.consumers]
                if diff_node == self.target_node:
                    diff_consumer_gradient = [1.0]
                    consumers = [(0, ValueNode(np.ones_like(self.target_node.get_value())))]
                else:
                    diff_consumer_gradient = self.compute_gradients(consumers, gradient_table)
                    consumers = diff_node.consumers

                diff_gradient = np.sum(
                    consumer_node.get_partial_derivative([inp.get_value() for inp in consumer_node.inputs],
                                                         target_consumer_gradient, arg_idx)
                    for (arg_idx, consumer_node), target_consumer_gradient
                    in zip(consumers, diff_consumer_gradient))

                gradient_table[diff_node] = diff_gradient

        return [gradient_table[target_node] for target_node in diff_nodes]
