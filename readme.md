# NN-Autograd

This is a simple implementation of automatic differentiation for neural networks. Neural networks can be defined as
computational graphs, which allows for the automatic computation of the network's gradients using back propagation.

## Simple example

```
layers = [DenseLayer(img_dim, 100), DenseLayer(100, 50), DenseLayer(50, 10)]
gradient_nodes = reduce(lambda a, b: a+b, (layer.get_trainable_values() for layer in layers))

a1 = layers[0](input)
h1 = gc.relu(a1)
a2 = layers[1](h1)
h2 = gc.relu(a2)
a3 = layers[2](h2)
loss = gc.softmax_cross_entropy(a3, ys)

backprop = BackProp(loss)
gradients = backprop.compute_gradients(gradient_nodes)
```

More detailed examples for training NNs can be found in `examples/`
