# MXNet Model Zoo

MXNet features fast implementations of most state-of-the-art models reported in the academic literature. Our Model Playground contains complete models, with python scripts, pre-trained weights as well as instructions on how to fine tune these models.  

## How to Contribute a Pre-Trained Model (and what to include)

Issue a Pull Request containing the following: 
* Gist Log
* .json model definition
* Model parameter file
* Readme file (details below)
 
Readme file should contain:
* Model Location, access instructions (wget)
* Confirmation the trained model meets published accuracy from original paper 
* Step by step instructions on how to use the trained model
* References to any other applicable docs or arxiv papers the model is based on

## [Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

Convolutional neural networks are the state-of-art architecture for many image and video processing problems. Some available datasets include:

* [ImageNet](http://image-net.org/): a large corpus of 1 million natural images, divided into 1000 categories.
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html): 60,000 natural images (32 x 32 pixels) from 10 categories.
* [PASCAL_VOC](http://host.robots.ox.ac.uk/pascal/VOC/): A subset of ImageNet images with object bounding boxes.
* [UCF101](http://crcv.ucf.edu/data/UCF101.php): 13,320 videos from 101 action categories.
* [Mini-Places2](http://6.869.csail.mit.edu/fa15/project.html): Subset of the Places2 dataset. Includes 100,000 images from 100 scene categories.
* ImageNet 11k
* [Places2](http://places2.csail.mit.edu/download.html): There are 1.6 million train images from 365 scene categories in the Places365-Standard, which are used to train the Places365 CNNs. There are 50 images per category in the validation set and 900 images per category in the testing set. Compared to the train set of Places365-Standard, the train set of Places365-Challenge has 6.2 million extra images, leading to totally 8 million train images for the Places365 challenge 2016. The validation set and testing set are the same as the Places365-Standard.



| Model Definition | Dataset | Model Weights | Research Basis | Contributors |
| --- | --- | --- | --- | --- |
| CaffeNet | ImageNet | |   [Krizhevsky, 2012](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) | @.. |
| Network in Network (NiN) | CIFAR-10 | |  [Lin et al.., 2014](https://arxiv.org/pdf/1312.4400v3.pdf) | |
| SqueezeNet | ImageNet | | [Iandola et al.., 2016](https://arxiv.org/pdf/1602.07360v4.pdf) | |
| VGG16 | ImageNet | | [Simonyan et al.., 2015](https://arxiv.org/pdf/1409.1556v6.pdf) | |
| VGG19 | ImageNet | | [Simonyan et al.., 2015](https://arxiv.org/pdf/1409.1556v6.pdf) | |
| Inception v3 w/BatchNorm | ImageNet | | [Szegedy et al.., 2015](https://arxiv.org/pdf/1512.00567.pdf) | |
| ResidualNet152 | ImageNet | | [He et al.., 2015](https://arxiv.org/pdf/1512.03385v1.pdf) | |
| Fast-RCNN | PASCAL VOC | | [Girshick, 2015](https://arxiv.org/pdf/1504.08083v2.pdf) | |
| Faster-RCNN | PASCAL VOC |  | [Ren et al..,2016](https://arxiv.org/pdf/1506.01497v3.pdf) | |
| Single Shot Detection (SSD) | PASCAL VOC | | [Liu et al.., 2016](https://arxiv.org/pdf/1512.02325v4.pdf) | |


## [Recursive Neural Networks (Including LSTMs)](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)

MXNet supports recurrent neural networks (RNNs), as well as Long short-term memory (LSTM) networks, and Gated Recurrent Units (GRU) networks. Some available datasets include:


* [Penn Treebank (PTB)](https://www.cis.upenn.edu/~treebank/): Text corpus with ~1 million words. Vocabulary is limited to 10,000 words. The task is predicting downstream words/characters.
* [Shakespeare](http://cs.stanford.edu/people/karpathy/char-rnn/): Complete text from Shakespeare’s works.
* [IMDB reviews](https://s3.amazonaws.com/text-datasets): 25,000 movie reviews, labeled as positive or negative
* [Facebook bAbI](https://research.facebook.com/researchers/1543934539189348): As set of 20 question & answer tasks, each with 1,000 training examples.
* [Flickr8k, COCO](http://mscoco.org/): Images with associated caption (sentences). Flickr8k consists of 8,092 images captioned by AmazonTurkers with ~40,000 captions. COCO has 328,000 images, each with 5 captions. The COCO images also come with labeled objects using segmentation algorithms.


| Model Definition | Dataset | Model Weights | Research Basis | Contributors |
| --- | --- | --- | --- | --- |
| LSTM - Image Captioning | Flickr8k, MS COCO | | [Vinyals et al.., 2015](https://arxiv.org/pdf/ 1411.4555v2.pdf) | @... |
| LSTM - Q&A System| bAbl | | [Weston et al.., 2015](https://arxiv.org/pdf/1502.05698v10.pdf) | |
| LSTM - Sentiment Analysis| IMDB | | [Li et al.., 2015](http://arxiv.org/pdf/1503.00185v5.pdf) | |


## [Generative Adversarial Networks](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf) 
| Model Definition | Dataset | Model Weights | Research Basis | Contributors |
| --- | --- | --- | --- | --- |
| DCGANs | ImageNet | | [Radford et al..,2016](https://arxiv.org/pdf/1511.06434v2.pdf) | @... |
| Text to Image Synthesis |MS COCO| | [Reed et al.., 2016](https://arxiv.org/pdf/1605.05396v2.pdf) | | 
| Deep Jazz	| | | [Deepjazz.io](https://deepjazz.io) | |



## Other Models

MXNet Supports a variety of model types beyond the canonical CNN and LSTM model types. These include deep reinforcement learning, linear models, etc.. Some available datasets and sources include:

* [Google News](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit): A text corpus with a vocabulary of 3 million words architected for word2vec.
* [MovieLens 20M Dataset](http://grouplens.org/datasets/movielens/): 20 million ratings and 465,000 tag applications applied to 27,000 movies by 138,000 users. Includes tag genome data with 12 million relevance scores across 1,100 tags.
* [Atari Video Game Emulator](http://stella.sourceforge.net/): Stella is a multi-platform Atari 2600 VCS emulator released under the GNU General Public License (GPL).

 
| Model Definition | Dataset | Model Weights | Research Basis | Contributors |
| --- | --- | --- | --- | --- |
| Word2Vec | Google News | | [Mikolov et al.., 2013](https://arxiv.org/pdf/1310.4546v1.pdf) | @... |
| Matrix Factorization | MovieLens 20M | | [Huang et al.., 2013](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf) | |
| Deep Q-Network | Atari video games | | [Minh et al.., 2015](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html) | |
| Asynchronous advantage actor-critic (A3C) | Atari video games | | [Minh et al.., 2016](https://arxiv.org/pdf/1602.01783.pdf) | |






