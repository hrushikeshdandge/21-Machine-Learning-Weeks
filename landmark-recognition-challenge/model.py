import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense,Flatten, Conv2D,MaxPooling2D
from keras.models import Sequential
import numpy as np
import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import pickle

pkl_file = open('data.pkl', 'rb')
dataset = pickle.load(pkl_file)
pkl_file.close()



#settings 
batch_size = 128
epochs = 40


x=dataset['x']
y=dataset['y']

x_train,x_val,y_train,y_val=train_test_split(x,y)
input_shape=[x_train.shape[1],x_train.shape[2],x_train.shape[3]]

print("X shape "+str(x.shape))
print("Y shape "+str(y.shape))
n_classes=y_train.shape[1]
n_samples=y_train.shape[0]

print(str(n_classes)+' number of classes')
print(str(n_samples)+' number of samples')

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_val, y_val),
          callbacks=[history])
score = model.evaluate(x_val, y_val, verbose=0)

print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

model.save('model.h5')


# test_	=pd.read_csv("data/test.csv")
# x_test_data=pca.transform(test_df.values)
# x_test_data=x_test_data.reshape([x_test_data.shape[0],25,25,1])

# val=[]

# for row in model.predict(x_test_data):
#   val.append(row.argmax(axis=0))
#   pass



# d = {'ImageId':range(1,x_test_data.shape[0]+1),'label': val}
# result_df = pd.DataFrame(data=d)
# result_df.reset_index(drop=True,inplace=True)

# result_df.to_csv('results.csv', sep=',', encoding='utf-8')













