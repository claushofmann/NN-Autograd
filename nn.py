import graph_creation_api as gc
import numpy as np
from abc import ABC, abstractmethod


class TrainableLayer(ABC):
    @abstractmethod
    def get_trainable_values(self):
        ...


class DenseLayer:
    def __init__(self, no_input, no_output):
        self.w = gc.value(np.random.uniform(-.5, .5, size=[no_input, no_output]))
        self.b = gc.value(np.zeros([no_output]))

    def __call__(self, inputs, debug=False):
        w = self.w
        b = self.b
        if debug:
            w = gc.debug(w)
            b = gc.debug(b)
        mul = gc.matmul(inputs, w)
        a = gc.add(mul, b)
        return a

    def get_trainable_values(self):
        return [self.w, self.b]