# Cloud Setup for MXNet

## Set Up an AWS GPU Cluster from Scratch

In this topic, we provide a step-by-step tutorial on how to set up an AWS cluster with
MXNet. We show how to:

- [Use Amazon S3 to host data](#use-amazon-s3-to-host-data)
- [Set up an EC2 GPU instance with all dependencies installed](#set-up-an-ec2-gpu-instance)
- [Build and run MXNet on a single computer](#build-and-run-mxnet-on-a-gpu-instance)
- [Set up an EC2 GPU cluster for distributed training](#set-up-an-ec2-gpu-cluster-for-distributed-training)

### Use Amazon S3 to Host Data

Amazon S3 provides distributed data storage, which is convenient for hosting large datasets. To use S3, you need [AWS credentials](http://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSGettingStartedGuide/AWSCredentials.html),
which include an `ACCESS_KEY_ID` and a `SECRET_ACCESS_KEY`.

To use MXNet with S3, set the environment variables `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` by adding the following two lines in
`~/.bashrc` (replacing the strings with the correct ones):

```bash
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

There are several ways to upload data to S3. A simple way is using
[s3cmd](http://s3tools.org/s3cmd). For example:

```bash
wget http://data.mxnet.io/mxnet/data/mnist.zip
unzip mnist.zip && s3cmd put t*-ubyte s3://dmlc/mnist/
```

### Set Up an EC2 GPU Instance

MXNet requires the following libraries:

- C++ compiler with C++11 support, such as `gcc >= 4.8`
- `CUDA` (`CUDNN` in optional) for GPU linear algebra
- `BLAS` (cblas, open-blas, atblas, mkl, or others) for CPU linear algebra
- `opencv` for image augmentations
- `curl` and `openssl` for the ability to read/write to Amazon S3

Installing `CUDA` on EC2 instances requires some effort. Caffe has a good
[tutorial](https://github.com/BVLC/caffe/wiki/Install-Caffe-on-EC2-from-scratch-(Ubuntu,-CUDA-7,-cuDNN))
on how to install CUDA 7.0 on Ubuntu 14.04.

***Note:*** We tried CUDA 7.5 on Nov 7,
2015, but found it problematic.

You can install the rest using the package manager. For example, on Ubuntu:

```
sudo apt-get update
sudo apt-get install -y build-essential git libcurl4-openssl-dev libatlas-base-dev libopencv-dev python-numpy
```

The Amazon Machine Image [ami-12fd8178](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#LaunchInstanceWizard:ami=ami-12fd8178) has the  packages listed above installed.


### Build and Run MXNet on a GPU Instance

The following commands build MXNet with CUDA/CUDNN, Amazon S3, and distributed
training.

```bash
git clone --recursive https://github.com/dmlc/mxnet
cd mxnet; cp make/config.mk .
echo "USE_CUDA=1" >>config.mk
echo "USE_CUDA_PATH=/usr/local/cuda" >>config.mk
echo "USE_CUDNN=1" >>config.mk
echo "USE_BLAS=atlas" >> config.mk
echo "USE_DIST_KVSTORE = 1" >>config.mk
echo "USE_S3=1" >>config.mk
make -j$(nproc)
```

To test whether everything is installed properly, train a convolutional neural network (CNN) on MNIST using a GPU:

```bash
python tests/python/gpu/test_conv.py
```

If you've placed the MNIST data on `s3://dmlc/mnist`, you can read the data stored on Amazon S3 directly with the following command:

```bash
sed -i.bak "s!data_dir = 'data'!data_dir = 's3://dmlc/mnist'!" tests/python/gpu/test_conv.py
```

***Note:*** You can use `sudo ln /dev/null /dev/raw1394` to fix the opencv error `libdc1394 error: Failed to initialize libdc1394`.

### Set Up an EC2 GPU Cluster for Distributed Training

A cluster consists of multiple computers. You can use the computer with MXNet
installed as the root computer for submitting jobs, and then launch several
slave computers to run the jobs. For example, launch multiple instances using an
AMI, e.g.,
[ami-12fd8178](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#LaunchInstanceWizard:ami=ami-12fd8178),
with dependencies installed. There are two options:

- Make all slaves' ports accessible (same for the root) by setting type: All TCP,
   Source: Anywhere in Configure Security Group.

- Use the same `pem` as the root computer to access all slave computers, and
   then copy the `pem` file into the root computer's `~/.ssh/id_rsa`. If you do this, all slave computers can be accessed with SSH from the root.

Now, run the CNN on multiple computers. Assume that we are on a working
directory of the root computer, such as `~/train`, and MXNet is built as `~/mxnet`.

1. Pack the MXNet Python library into this working directory for easy
  synchronization:

  ```bash
  cp -r ~/mxnet/python/mxnet .
  cp ~/mxnet/lib/libmxnet.so mxnet/
  ```

  And then copy the training program:

  ```bash
  cp ~/mxnet/example/image-classification/*.py 
  ```

2. Prepare a host file with all slaves's private IPs. For example, `cat hosts`:

  ```bash
  172.30.0.172
  172.30.0.171
  ```

3. Assuming that there are two computers, train the CNN using two workers:

  ```bash
  ../../tools/launch.py -n 2 -H hosts --sync-dir /tmp/mxnet python train_mnist.py --kv-store dist_sync
  ```

***Note:*** Sometimes the jobs linger at the slave computers even you've pressed `Ctrl-c`
at the root node. To terminate them, use the following command:

```bash
cat hosts | xargs -I{} ssh -o StrictHostKeyChecking=no {} 'uname -a; pgrep python | xargs kill -9'
```

***Note:*** The preceding example is very simple to train, and, therefore, isn't a good
benchmark for distributed training. Consider using other [examples](https://github.com/dmlc/mxnet/tree/master/example/image-classification).

### More Options
#### Use Multiple Data Shards
It is common to pack a dataset into multiple files, especially when working in a distributed environment. MXNet supports direct loading from multiple data shards. Pput all of the record files into a folder, and point the data path to the folder.

#### Use YARN and SGE
Although using SSH can be simple when you don't have a cluster scheduling framework, MXNet is designed to be portable to various platforms.  We provide other scripts in [tracker](https://github.com/dmlc/dmlc-core/tree/master/tracker) to allow running on other cluster frameworks, including Hadoop (YARN) and SGE. We welcome your contribution of examples of running MXNet on your favorite distributed platform.
