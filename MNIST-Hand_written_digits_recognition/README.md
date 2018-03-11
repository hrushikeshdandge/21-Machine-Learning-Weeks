# DeepLearningModels

<p>Before running this code, make sure you install tensorflow and sklearn<p>

<u href="https://www.tensorflow.org/install/">Install Tensor Flow</u>


<p><b>Note</b></p>
You can train this neural network on any custom dataset (either from csv files using pandas or any sklearn dataset)
<b>data_frame</b> will be your dataframe.
whatsover you have to adjust number of neurons (n_hidden_1, n_hidden_2) in hidden layers according to number of features and labels. 

Example :
input (n_features): 1000
n_hidden_1 : 700
n_hidden_2 : 300
output (n_classes): 100



Neural Network with 2 hidden layers taking input as handwritten dataset(15K*64) and mapping it to 10 classes

Loss Function : Cross Entropy
Optimizer of Loss Function : Adam Optimizer

Neural Network Structure
<ul>

	<li>input: 64 ( number of features )</li>
	<li>hidden_layer_1 : 40</li>
    <li>hidden_layer_2 : 20</li>
    <li>output: 10 (number of classes)</li>

</ul>

<h2>Weights and Biases</h2>
<img src="Images/histogram.png"/>

<h2>Accuracy and Cost<h2>
<img src="Images/scaler.png"/>


To visualize flow, install tensorboard and run this script,then 

run '''tensorboard --logdir=tmp/mnist'''
