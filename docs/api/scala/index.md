# MXNet - Scala API

See the [MXNet Scala API Documentation](http://mxnet.io/api/scala/docs/index.html).

MXNet supports the Scala programming language. The MXNet Scala package brings flexible and efficient GPU
computing and state-of-art deep learning to Scala. It enables you to write seamless tensor/matrix computation with multiple GPUs in Scala. It also lets you construct and customize the state-of-art deep learning models in Scala,
  and apply them to tasks, such as image classification and data science challenges.

You can perform tensor or matrix computation in pure Scala:

 ```scala
    scala> import ml.dmlc.mxnet._
    import ml.dmlc.mxnet._

    scala> val arr = NDArray.ones(2, 3)
    arr: ml.dmlc.mxnet.NDArray = ml.dmlc.mxnet.NDArray@f5e74790

    scala> arr.shape
    res0: ml.dmlc.mxnet.Shape = (2,3)

    scala> (arr * 2).toArray
    res2: Array[Float] = Array(2.0, 2.0, 2.0, 2.0, 2.0, 2.0)

    scala> (arr * 2).shape
    res3: ml.dmlc.mxnet.Shape = (2,3)
 ```

## Resources

* [MXNet Scala API Documentation](http://mxnet.io/api/scala/docs/index.html)
* [Handwritten Digit Classification in Scala](http://mxnet.io/tutorials/scala/mnist.html)
* [Neural Style in Scala on MXNet](https://github.com/dmlc/mxnet/blob/master/scala-package/examples/src/main/scala/ml/dmlc/mxnet/examples/neuralstyle/NeuralStyle.scala)
* [More Scala Examples](https://github.com/dmlc/mxnet/tree/master/scala-package/examples/src/main/scala/ml/dmlc/mxnet/examples)
