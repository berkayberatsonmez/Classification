#1.kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

#2.verionisleme
#2.1 veri yukleme
veriler = pd.read_excel("veriler.xlsx")

print(veriler)

x= veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33,random_state=0)

#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train) 
X_test = sc.transform(x_test) #fit eğitme, transform ise o öğrendiği eğitimi kullanma işlemi demektir


from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state = 0) 
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)

#confusion-matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
print(cm)







