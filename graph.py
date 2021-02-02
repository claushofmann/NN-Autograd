from ops import Op
from abc import ABC, abstractmethod
import numpy as np

class Node(ABC):
    def __init__(self):
        self.consumers = []

    @abstractmethod
    def get_value(self):
        ...

    @abstractmethod
    def is_dirty(self):
        ...

    def add_consumer(self, index, consumer):
        self.consumers.append((index, consumer))

    @abstractmethod
    def get_partial_derivative(self, inputs, output_gradient, input_index):
        ...

    @abstractmethod
    def visit(self, visitor):
        ...

    @abstractmethod
    def _local_invalidate(self, invalidator):
        ...

    def _invalidate(self, invalidator):
        self._local_invalidate(invalidator)
        for i, c in self.consumers:
            c._invalidate(self)


class ComputedNode(Node):
    def __init__(self, op, inputs):
        super().__init__()
        assert(len(inputs) == op.no_inputs)
        self.op = op
        self.inputs = inputs
        self.cache = None
        for i, inp in enumerate(self.inputs):
            inp.add_consumer(i, self)

    def is_dirty(self):
        return self.cache is None

    def get_value(self):
        if self.is_dirty() or any(inp.is_dirty() for inp in self.inputs):
            self.cache = self.op.op([inp.get_value() for inp in self.inputs])
        return self.cache

    def get_partial_derivative(self, inputs, output_gradient, input_index):
        return self.op.bprop(inputs, output_gradient, input_index)

    def visit(self, visitor, *args):
        return visitor.visit_computed_node(self, *args)

    def _local_invalidate(self, invalidator):
        self.cache = None


class ValueNode(Node):
    def __init__(self, value):
        super().__init__()
        self.__value = value
        self.dirty = True
        self.inputs = []

    def is_dirty(self):
        return self.dirty

    def set_value(self, value):
        assert value.shape == self.__value.shape
        self.__value = value
        self._invalidate(self)

    def get_value(self):
        return self.__value

    def get_partial_derivative(self, inputs, output_gradient, input_index):
        return output_gradient

    def visit(self, visitor, *args):
        return visitor.visit_value_node(self, *args)

    def _local_invalidate(self, invalidator):
        self.dirty = True


class NodeVisitor(ABC):
    @abstractmethod
    def visit_value_node(self, node, *args):
        ...

    @abstractmethod
    def visit_computed_node(self, node, *args):
        ...