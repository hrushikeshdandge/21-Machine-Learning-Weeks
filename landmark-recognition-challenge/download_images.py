import numpy as np
import pandas as pd
import requests
import os
#importing training data
train_df=pd.read_csv("data/train.csv")
print(train_df.values.shape)


def download_images():

	try:
		count=0

		for i in train_df.values:

			label_path='images/'+str(i[2])+'/'
			if not os.path.exists(label_path):
   		 		os.makedirs(label_path)

			f = open(label_path+str(i[0])+'.jpg','wb')
			f.write(requests.get(i[1]).content)
			f.close()
			print(str(count)+'/'+str(train_df.values.shape[0]))
			count=count+1
			pass

		
	except Exception as e:
		print(e)
		pass
	except Error as er:
		print(er)

	pass

#preprocessing data

download_images()

















