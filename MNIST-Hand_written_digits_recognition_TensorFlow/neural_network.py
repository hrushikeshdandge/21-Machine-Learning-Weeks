""" Multilayer Perceptron.

A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library.

Author: SaiTej Dandge
Project: https://github.com/saitejdandge/MachineLearning

"""

from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

#Gives One-Hot Encoded matrix for all labels
def get_sparse_matrix(list_value):

    list_value=list_value.tolist()
    set_value=set(list_value)
    
    outer=[]
    k=0
    total=len(list_value)*len(set_value)
    for i in list_value:
        
        inner=[]

        percent=((k/total)*100)
        print(percent)
        print("Parsing Output to One-Hot: ", percent)
        
        for j in set_value:

            if i == j:
                inner.append(1)
            else :
                inner.append(0)
                pass

            k+=1
            pass
        outer.append(inner)
        
        pass

    return np.array(outer)

    pass

#Preprocess data
def preprocess(data_frame):
    
    features_count=data_frame.data.shape[1]
    
    sparsed_target=get_sparse_matrix(data_frame.target)
    
    labels_count=sparsed_target.shape[1]

    return features_count,labels_count,sparsed_target

#Returns batch from training sample of given batch_size
def get_batch(X_train,batch_size):

    features_batch=[]

    labels_batch=[]

    random_index=np.random.randint(0,X_train.shape[0]-1)


    for i in range(0,batch_size):

        features_batch.append(X_train[random_index])
        
        labels_batch.append(y_train[random_index])
        pass


    return features_batch,labels_batch
    
    pass

import matplotlib.pyplot as plt
def plot(x,y,epoch,avg_cost):

  plt.scatter(x,y,color='blue')
  plt.plot(epoch,avg_cost,C='blue')
  plt.xlabel("Epoches")
  plt.ylabel("Avg Cost")
  plt.pause(0.0000005)
  pass



#importing dataset from sklearn
from sklearn.datasets import load_digits

data_frame=load_digits()

n_input,n_classes,sparsed_target= preprocess(data_frame)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(data_frame.data,sparsed_target)



import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 500 
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 40 # 1st layer number of neurons
n_hidden_2 = 20 # 2nd layer number of neurons

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}



print("########## Data Info #############")
print(X_train.shape)
print(n_input)
print(n_classes)
print("########## Data Info #############")


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# # Construct model
# logits = multilayer_perceptron(X)

# # Define loss and optimizer
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     logits=logits, labels=Y))

# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)


# train_op = optimizer.minimize(loss_op)
# # Initializing the variables
# tf.summary.scalar('cost',loss_op)




with tf.name_scope('Model'):
    # Model
    pred = multilayer_perceptron(X) # Softmax
with tf.name_scope('Loss'):
    # Minimize error using cross entropy
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=Y))

with tf.name_scope('AdamOptimizer'):
    # Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
with tf.name_scope('Accuracy'):
    # Accuracy
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))

train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()



# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)




tf.summary.histogram("w_h1",weights['h1'])
tf.summary.histogram("w_h2",weights['h2'])
tf.summary.histogram("w_out",weights['out'])

tf.summary.histogram("b_h1",biases['b1'])
tf.summary.histogram("b_h2",biases['b2'])
tf.summary.histogram("b_out",biases['out'])

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

k = tf.placeholder(tf.float32)

# Make a normal distribution, with a shifting mean
mean_moving_normal = tf.random_normal(shape=[1000], mean=(5*k), stddev=1)
# Record that distribution into a histogram summary
tf.summary.histogram("normal/moving_mean", mean_moving_normal)

with tf.Session() as sess:
    
    sess.run(init)

    # Training cycle


    epoches=[]
    avg_costs=[]

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter('tmp/mnist')

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(X_train.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = get_batch(X_train,batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c , summary= sess.run([train_op, cost,merged_summary_op], feed_dict={X: batch_x, Y: batch_y})
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            epoches.append(epoch+1)
            avg_costs.append(avg_cost)
            print("-------------------")
           
            #plot(epoch+1,avg_cost,epoches,avg_costs)


    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(pred)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
 
    print("Accuracy:", accuracy.eval({X: X_test, Y: y_test}))



















