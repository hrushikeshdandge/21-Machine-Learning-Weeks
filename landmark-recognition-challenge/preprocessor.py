#converting downloaded images to 2D arrays
from scipy.misc import imresize
from imageio import imread
import os
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Dense,Flatten, Conv2D,MaxPooling2D
from keras.models import Sequential
import numpy as np
import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import pickle
x=[]
y=[]

rootdir = 'images'

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def convert(path):

	try:
		return imresize(imread(path), (150, 150))
	except Exception as e:
		print(e)
	except Error as er:
		print(er)
	pass

count=0

for subdir in next(os.walk(rootdir+'/.'))[1]:
	
	for file in os.listdir(rootdir+'/'+subdir):
		if not file.startswith('.'):   #ignoring hidden files
			val=convert(rootdir+'/'+subdir+'/'+file)
			# print(val.shape)
			x.append(val)

			y.append(subdir)
			print('progress '+str(count))
			count=count+1
			pass	
		pass
	pass


x=np.array(x)
le =LabelEncoder()
le.fit(y)
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)
y=le.transform(y)

val=[]
for i in y:
	val.append([i])
	pass
y=val




ohc=OneHotEncoder(sparse=False)
y=ohc.fit_transform(y)



import pickle

print(x.shape)
print(y.shape)
my_data = {'x': x,'y': y,'ohc':ohc,'le':le,'le_map':le_name_mapping}
output = open('data.pkl', 'wb')

pickle.dump(my_data, open("data.pkl", 'wb'), protocol=4)
output.close()

