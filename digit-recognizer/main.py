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


df=pd.read_csv("data/train.csv")

#settings 
batch_size = 128
epochs = 40
#perfect square
no_pca_components=625
pca=PCA(n_components=no_pca_components)
ohc=OneHotEncoder(sparse=False)
np.set_printoptions(threshold=np.nan)
def preprocess(x_df):
 
  x_data=x_df.values
  x_data=pca.fit_transform(x_data)
  x_data=x_data.reshape([42000,25,25,1])
  return x_data


x_df=df.drop(['label'],axis=1)
x_data=preprocess(x_df)



y_df=df[['label']]
y_data=y_df.values

print("-----------------------  After Encoding -----------------------")
y_data=ohc.fit_transform(y_data)

print("-----------------------  X shape -----------------------")
print(x_data.shape)
print("-----------------------  Y shape -----------------------")
print(y_data.shape)



input_shape=[x_data.shape[1],x_data.shape[2],x_data.shape[3]]
x_train,x_val,y_train,y_val=train_test_split(x_data,y_data)


model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))

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

model.save('digits_cnn.h5')


test_df=pd.read_csv("data/test.csv")
x_test_data=pca.transform(test_df.values)
x_test_data=x_test_data.reshape([x_test_data.shape[0],25,25,1])

val=[]

for row in model.predict(x_test_data):
  val.append(row.argmax(axis=0))
  pass



d = {'ImageId':range(1,x_test_data.shape[0]+1),'label': val}
result_df = pd.DataFrame(data=d)
result_df.reset_index(drop=True,inplace=True)

result_df.to_csv('results.csv', sep=',', encoding='utf-8')













