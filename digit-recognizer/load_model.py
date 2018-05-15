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

model = load_model('my_model.h5')



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
