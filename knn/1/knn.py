import os
import pandas as pd
from sklearn.model_selection import train_test_split #引入train_test_split函数
from sklearn import preprocessing  as pp
from sklearn import neighbors
from sklearn import datasets
from sklearn.model_selection import cross_val_score
os.chdir(r"E:\MachineLearning\knn\1")
data=pd.read_csv('ecoli.csv')
#输入特征值 Mcg, Gvh, Lip, Chg, Aac, Alm1, Alm2
#输出 Site  {cp,im,imS,imL,imU,om,omL,pp}
#数据处理   添加表头属性值
data.columns=["ID","Mcg","Gvh","Lip","Chg","Aac","Alm1","Alm2","Site"]
data.isnull().sum()    #查看是否有缺失值
data.shape    


le=pp.LabelEncoder() #label encoder就是把lable编码的成数字便于训练  
le.fit(data.Site)  


knn=neighbors.KNeighborsClassifier(6,weights='distance')  #定义knn

x=data[data.columns[1:7]]
y=le.transform(data.Site)  

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3) 


knn.fit(X_train,y_train)    #训练KNN分类器
print(knn.predict(X_test))  #预测值
print(y_test) 

 
 