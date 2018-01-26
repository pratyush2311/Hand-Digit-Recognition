import numpy as np
import matplotlib.pyplot as pt
import pandas as pd  
from  sklearn.tree import DecisionTreeClassifier
data=pd.read_csv("train.csv").as_matrix()


#classifier
clf=DecisionTreeClassifier()


#training data
xtrain=data[0:21000,1:]
train_label=data[0:21000,0]


clf.fit(xtrain,train_label)

#testing data
xtest=data[21000:,1:]
actual_label=data[21000:,0]

"""
Random testing of a test data
d=xtest[8]
d.shape=(28,28)
pt.imshow(255-d,cmap='gray')
pt.show()

print(clf.predict([xtest[8]]))
"""

p=clf.predict(xtest)

count=0
for i in range(0,21000):
	if(p[i]==actual_label[i]):
		count+=1


print("Accuracy=",(count/21000.0)*100)



