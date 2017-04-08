# How to visualize Neural Networks as computation graph

This topic demonstrates how to use ```mx.viz.plot_network``` in MXNet for visualizing your Neural Networks built on MXNet. ```mx.viz.plot_network``` helps to represent the Neural Network as a computation graph of nodes; with input nodes, where the computation starts, and output nodes, where the result can be read.

## Prerequisites
You need [Jupyter Notebook](jupyter.readthedocs.io) and [Graphviz](http://www.graphviz.org/) library to visualize the network. Please Make sure you have followed [installation instructions](http://mxnet.io/get_started/setup.html) in setting up above dependencies along with setting up MXNet.

## Visualize the sample Neural Network

```mx.viz.plot_network``` takes [Symbol](http://mxnet.io/api/python/symbol.html), with your Network definition, and optional node_attrs, parameters for shape of node in the graph,  as input and generates a computation graph.

We will now try to visualize a sample Neural Network for linear matrix factorization:
- Start Jupyter notebook server
```bash
  $ jupyter notebook
```
- Aceess Jupyter Notebook in your browser - http://localhost:8888/.
- Create a new notebook - "File -> New Notebook -> Python 2"
- Copy and and run below code to visualize a sample network.

```python
  import mxnet as mx
  user = mx.symbol.Variable('user')
  item = mx.symbol.Variable('item')
  score = mx.symbol.Variable('score')

  # Set dummy dimensions
  k = 64
  max_user = 100
  max_item = 50

  # user feature lookup
  user = mx.symbol.Embedding(data = user, input_dim = max_user, output_dim = k)

  # item feature lookup
  item = mx.symbol.Embedding(data = item, input_dim = max_item, output_dim = k)

  # predict by the inner product, which is elementwise product and then sum
  net = user * item
  net = mx.symbol.sum_axis(data = net, axis = 1)
  net = mx.symbol.Flatten(data = net)

  # loss layer
  net = mx.symbol.LinearRegressionOutput(data = net, label = score)

  # Visualize your network
  mx.viz.plot_network(net)
```
You should be able to see computation graph something like below:
<img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/SampleNetworkVisualization.png
width=400/>

# References
* [Example MXNet Matrix Factorization](https://github.com/dmlc/mxnet/blob/master/example/recommenders/demo1-MF.ipynb)
* [Visualizing CNN Architecture of MXNet Tutorials](http://josephpcohen.com/w/visualizing-cnn-architectures-side-by-side-with-mxnet/)
