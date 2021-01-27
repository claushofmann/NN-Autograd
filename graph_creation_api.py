from graph import Node, ComputedNode, ValueNode
import ops
import numpy as np


def matmul(A:Node, B:Node):
    return ComputedNode(ops.Matmul(), [A, B])

def relu(inp: Node):
    return ComputedNode(ops.Relu(), [inp])

def softmax_cross_entropy(a: Node, y: Node):
    return ComputedNode(ops.SoftmaxCrossEntropyLoss(), [a, y])

def add(A:Node, B:Node):
    return ComputedNode(ops.Add(), [A, B])

def multiply(A:Node, B:Node):
    return ComputedNode(ops.Multiply(), [A, B])

def negative(inp:Node):
    return ComputedNode(ops.Negative(), [inp])

def value(val: np.array):
    return ValueNode(val)

def debug(n: Node):
    return ComputedNode(ops.Debug(), [n])