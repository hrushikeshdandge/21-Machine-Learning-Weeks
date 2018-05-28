Many Kagglers are familiar with image classification challenges like the ImageNet Large Scale Visual Recognition Challenge (ILSVRC), which aims to recognize 1K general object categories. Landmark recognition is a little different from that: it contains a much larger number of classes (there are a total of 15K classes in this challenge), and the number of training examples per class may not be very large. Landmark recognition is challenging in its own way.

This challenge is organized in conjunction with the Landmark Retrieval Challenge (https://www.kaggle.com/c/landmark-retrieval-challenge). In particular, note that the test set for both challenges is the same, to encourage participants to compete in both. We also encourage participants to use the training data from the recognition challenge to train models which could be useful for the retrieval challenge. Note, however, that there are no landmarks in common between the training/index sets of the two challenges.