import numpy as np
from abc import ABC, abstractmethod


class Op(ABC):
    def __init__(self, no_inputs):
        self.no_inputs = no_inputs

    def op(self, inputs):
        assert(len(inputs) == self.no_inputs)
        return self._op(inputs)

    @abstractmethod
    def _op(self, inputs):
        ...

    def bprop(self, inputs, output_gradient, input_index):
        assert(len(inputs) == self.no_inputs)
        assert(input_index < self.no_inputs)
        grad = self._bprop(inputs, output_gradient, input_index)
        assert grad.shape == inputs[input_index].shape
        return grad

    @abstractmethod
    def _bprop(self, inputs, output_gradient, input_index):
        ...


class Matmul(Op):
    def __init__(self):
        super().__init__(2)

    def _op(self, inputs):
        A, B = inputs
        return np.matmul(A, B)

    def _bprop(self, inputs, output_gradient, input_index):
        A, B = inputs
        if input_index == 0:
            # Gradient w.r.t. A
            return np.matmul(output_gradient, np.transpose(B))
        if input_index == 1:
            # Gradient w.r.t. B
            return np.matmul(np.transpose(A), output_gradient)


class Relu(Op):
    def __init__(self):
        super().__init__(1)

    def _op(self, inputs):
        A, = inputs
        return np.maximum(np.zeros_like(A), A)

    def _bprop(self, inputs, output_gradient, input_index):
        A, = inputs
        relu_gradient = (A >= 0).astype(A.dtype)
        return np.multiply(relu_gradient, output_gradient)


def broadcasting_bprop_helper(inputs, input_index):
    A, B = inputs
    # We have to consider numpy broadcasting, therefore the shape stuff
    if input_index == 0:
        gradient_shape = list(A.shape)
        other_shape = list(B.shape)
    elif input_index == 1:
        gradient_shape = list(B.shape)
        other_shape = list(A.shape)

    if len(gradient_shape) != len(other_shape):
        smaller_shape = min([gradient_shape, other_shape], key=len)
        larger_shape = max([gradient_shape, other_shape], key=len)
        while len(smaller_shape) < len(larger_shape):
            smaller_shape.insert(0, 1)
    # incompatible shapes will be recognized in forward prop

    sum_axes = tuple(i for i, (gradient_size) in enumerate(gradient_shape) if gradient_size == 1)
    # if other size is also 1, sum over this axis will do nothing

    return sum_axes


class Multiply(Op):
    def __init__(self):
        super().__init__(2)

    def _op(self, inputs):
        A, B = inputs
        return np.multiply(A, B)

    def _bprop(self, inputs, output_gradient, input_index):
        if input_index == 0:
            grad = np.multiply(inputs[1] * output_gradient)
        elif input_index == 1:
            grad = np.multiply(inputs[0] * output_gradient)

        sum_axes = broadcasting_bprop_helper(inputs, input_index)

        if sum_axes:
            return_gradient = np.sum(grad, axis=sum_axes)
        else:
            return_gradient = grad

        return return_gradient


class Add(Op):
    def __init__(self):
        super().__init__(2)

    def _op(self, inputs):
        A, B = inputs
        return np.add(A, B)

    def _bprop(self, inputs, output_gradient, input_index):
        sum_axes = broadcasting_bprop_helper(inputs, input_index)

        if sum_axes:
            return_gradient = np.sum(output_gradient, axis=sum_axes)
        else:
            return_gradient = output_gradient

        return return_gradient


class Negative(Op):
    def __init__(self):
        super().__init__(1)

    def _op(self, inputs):
        A, = inputs
        return np.negative(A)

    def _bprop(self, inputs, output_gradient, input_index):
        return np.negative(output_gradient)


class SoftmaxCrossEntropyLoss(Op):
    def __init__(self):
        super().__init__(2)

    def _op(self, inputs):
        A, y = inputs
        exp = np.exp(A)
        softmax = np.divide(exp, np.sum(exp, axis=1, keepdims=True))

        cross_entropy = -np.log(softmax[np.array(range(len(y))), y])

        return np.mean(cross_entropy)

    def _bprop(self, inputs, output_gradient, input_index):
        if input_index == 1:
            # gradient w.r.t. y
            raise NotImplementedError()
        if input_index == 0:
            A, y = inputs
            exp = np.exp(A)
            softmax = np.divide(exp, np.sum(exp, axis=1, keepdims=True))

            # y_l = list(enumerate(y))
            gradient = softmax
            gradient[np.array(range(len(y))), y] -= 1

            return output_gradient * gradient


class Slice(Op):
    def __init__(self):
        super().__init__(2)

    def _op(self, inputs):
        A, slicing = inputs
        return np.array(A).__getitem__(slicing)

    def _bprop(self, inputs, output_gradient, input_index):
        A, slicing = inputs

        grad = np.zeros_like(A)
        grad.__setitem__(slicing, output_gradient)

        return grad

class Debug(Op):
    # Can be used to place breakpoints at specific locations in the computational graph
    def __init__(self):
        super().__init__(1)

    def _op(self, inputs):
        return inputs[0]

    def _bprop(self, inputs, output_gradient, input_index):
        return output_gradient


if __name__ == '__main__':
    ce = SoftmaxCrossEntropyLoss()
    A = np.array([[0, 1, 2], [3, 4, 5]])
    y = [2, 1]
    output = ce.op([A, y])
    output_gradients = np.ones_like(output)
    gradient = ce.bprop([A, y], output_gradients, 0)
    print(output)
    print(gradient)