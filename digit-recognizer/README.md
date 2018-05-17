Kaggle Problem :
<a href="https://www.kaggle.com/c/digit-recognizer">https://www.kaggle.com/c/digit-recognizer</a>

Competition Description
MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

In this competition, your goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. We’ve curated a set of tutorial-style kernels which cover everything from regression to neural networks. We encourage you to experiment with different algorithms to learn first-hand what works well and how techniques compare.

Preprocessing :
* Dimensionality reduction using PCA to have perfect square dimension
   784    -----> 625
* Reshaping linear data as 2 * 2 array [42000 (number of samples) ,25,25,1 (number of channesl)]

*Model
    CNN network used.
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    conv2d_7 (Conv2D)            (None, 28, 28, 32)        2432
    _________________________________________________________________
    max_pooling2d_7 (MaxPooling2 (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 10, 10, 64)        51264
    _________________________________________________________________
    max_pooling2d_8 (MaxPooling2 (None, 5, 5, 64)          0
    _________________________________________________________________
    flatten_4 (Flatten)          (None, 1600)              0
    _________________________________________________________________
    dense_7 (Dense)              (None, 1000)              1601000
    _________________________________________________________________
    dense_8 (Dense)              (None,  100)              100100
    
    =================================================================
    
    Total params: 1,754,796
    Trainable params: 1,754,796
    Non-trainable params: 0
    

