# Titanic-Dataset---Survival-Prediction
Implementing different ML techniques to improve the prediction results.




# Survival Predication - Binary Classification using Different ML techniques
<p align="right">Prabhat Turlapati</p>

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Import-the-packages-required" data-toc-modified-id="Import-the-packages-required-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Import the packages required</a></span></li><li><span><a href="#Load-the-Data" data-toc-modified-id="Load-the-Data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Load the Data</a></span></li><li><span><a href="#Data-Cleaning,-Massaging-and-Engineering" data-toc-modified-id="Data-Cleaning,-Massaging-and-Engineering-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Data Cleaning, Massaging and Engineering</a></span></li><li><span><a href="#Building-the-Model-and-Testing-the-Accuracy" data-toc-modified-id="Building-the-Model-and-Testing-the-Accuracy-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Building the Model and Testing the Accuracy</a></span></li></ul></div>

## Import the packages required


```python
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier

# Use Keras simple neural network for prediction
from keras.models import Sequential
from keras.layers import Dense

# fix random seed for reproducibility
np.random.seed(7)

```

## Load the Data


```python
train_raw = pd.read_csv("train.csv")
test_raw = pd.read_csv("test.csv")
```


```python
from IPython.display import Image
Image("picture.png")
```




![png](output_6_0.png)



## Data Cleaning, Massaging and Engineering


```python
# remove un-necessary columns ticket,fare,cabin number,

columns =['PassengerId','Ticket','Fare','Cabin','Name','Embarked']
# train.info()
train = train_raw.drop(columns=columns)
test = test_raw.drop(columns=columns)

```


```python
train.head()
train = pd.get_dummies(data = train, columns=['Sex'])
# train = pd.get_dummies(data = train, columns=['Embarked'])
train = pd.get_dummies(data = train, columns=['Pclass'])

train.drop(columns=['Sex_female'],inplace=True)

test = pd.get_dummies(data = test, columns=['Sex'])
# test = pd.get_dummies(data = test, columns=['Embarked'])
test = pd.get_dummies(data = test, columns=['Pclass'])

test.drop(columns=['Sex_female'],inplace=True)
```


```python
print(train.info())
print(test.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 8 columns):
    Survived    891 non-null int64
    Age         714 non-null float64
    SibSp       891 non-null int64
    Parch       891 non-null int64
    Sex_male    891 non-null uint8
    Pclass_1    891 non-null uint8
    Pclass_2    891 non-null uint8
    Pclass_3    891 non-null uint8
    dtypes: float64(1), int64(3), uint8(4)
    memory usage: 31.4 KB
    None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 7 columns):
    Age         332 non-null float64
    SibSp       418 non-null int64
    Parch       418 non-null int64
    Sex_male    418 non-null uint8
    Pclass_1    418 non-null uint8
    Pclass_2    418 non-null uint8
    Pclass_3    418 non-null uint8
    dtypes: float64(1), int64(2), uint8(4)
    memory usage: 11.5 KB
    None
    


```python
print(train.loc[np.isnan(train['Age'])].head(5))
```

        Survived  Age  SibSp  Parch  Sex_male  Pclass_1  Pclass_2  Pclass_3
    5          0  NaN      0      0         1         0         0         1
    17         1  NaN      0      0         1         0         1         0
    19         1  NaN      0      0         0         0         0         1
    26         0  NaN      0      0         1         0         0         1
    28         1  NaN      0      0         0         0         0         1
    


```python
# Imputing with mean for age
train.Age = train.Age.fillna(value=train.Age.mean())
test.Age = test.Age.fillna(value=test.Age.mean())

```


```python
print(train.head())
print(test.head())
```

       Survived   Age  SibSp  Parch  Sex_male  Pclass_1  Pclass_2  Pclass_3
    0         0  22.0      1      0         1         0         0         1
    1         1  38.0      1      0         0         1         0         0
    2         1  26.0      0      0         0         0         0         1
    3         1  35.0      1      0         0         1         0         0
    4         0  35.0      0      0         1         0         0         1
        Age  SibSp  Parch  Sex_male  Pclass_1  Pclass_2  Pclass_3
    0  34.5      0      0         1         0         0         1
    1  47.0      1      0         0         0         0         1
    2  62.0      0      0         1         0         1         0
    3  27.0      0      0         1         0         0         1
    4  22.0      1      1         0         0         0         1
    


```python
y_train = train.Survived
X_train = train.drop(columns=['Survived'])

X_test = test
y_test_raw = pd.read_csv("gender_submission.csv")
y_test = y_test_raw.drop(columns='PassengerId')
```

## Building the Model and Testing the Accuracy


```python
# Use KNN 
knn_clf = KNeighborsClassifier()
gsv_knn = GridSearchCV(knn_clf,{'n_neighbors':[1,3,5]},cv=3,n_jobs=-1)
gsv_knn.fit(X_train,y_train)
y_pred_knn = gsv_knn.predict(X_test)
acc_knn = accuracy_score(y_pred_knn,y_test)
print("knn acc: ",acc_knn)

# Use DT
dt_clf = DecisionTreeClassifier()
gsv_dt = GridSearchCV(estimator=dt_clf,cv=3,n_jobs=-1,param_grid={'max_depth':[10,100,200,250,500,1000]})
gsv_dt.fit(X_train,y_train)
y_pred_dt = gsv_dt.predict(X_test)
acc_dt = accuracy_score(y_pred_dt,y_test)
print("dt acc: ",acc_dt)

# Use RF DT
rf_clf = RandomForestClassifier()
gsv_rf = GridSearchCV(estimator=rf_clf,cv=3,n_jobs=-1,param_grid={'n_estimators':[10,50,100,1000,10000]})
gsv_rf.fit(X_train,y_train)
y_pred_rf = gsv_rf.predict(X_test)
acc_rf = accuracy_score(y_pred_rf,y_test)
print("rf acc: ",acc_rf)



```

    knn acc:  0.7488038277511961
    dt acc:  0.7990430622009569
    rf acc:  0.777511961722488
    


```python
# Use SVC
svc_clf = SVC(kernel='linear')
gsv_svc = GridSearchCV(estimator=svc_clf,cv=3,n_jobs=-1,param_grid={'C':np.logspace(-4,0,1, 2, 3)})
gsv_svc.fit(X_train,y_train)
y_pred_svc = gsv_svc.predict(X_test)
acc_svc = accuracy_score(y_pred_svc,y_test)
print("svc acc: ",acc_svc)

```

    svc acc:  0.9976076555023924
    


```python
# Use Extremely Randomized forest Classifier
etc_clf = ExtraTreesClassifier(criterion='gini')
gsv_etc = GridSearchCV(estimator=etc_clf,cv=3,n_jobs=-1,param_grid={'n_estimators':[10,50,100,1000,10000]})
gsv_etc.fit(X_train,y_train)
y_pred_etc = gsv_etc.predict(X_test)
acc_etc = accuracy_score(y_pred_etc,y_test)
print("etc acc: ",acc_etc)

```

    etc acc:  0.7607655502392344
    


```python

model = Sequential()
model.add(Dense(12, input_dim=len(X_train.columns), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```


```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```


```python
model.fit(X_train, y_train, nb_epoch=300, batch_size=30)
```

    F:\Anaconda3\lib\site-packages\ipykernel_launcher.py:1: UserWarning: The `nb_epoch` argument in `fit` has been renamed `epochs`.
      """Entry point for launching an IPython kernel.
    

    Epoch 1/300
    891/891 [==============================] - 0s 368us/step - loss: 1.7415 - acc: 0.6162
    Epoch 2/300
    891/891 [==============================] - 0s 35us/step - loss: 0.9975 - acc: 0.6173
    Epoch 3/300
    891/891 [==============================] - 0s 35us/step - loss: 0.7588 - acc: 0.6229
    Epoch 4/300
    891/891 [==============================] - 0s 35us/step - loss: 0.6652 - acc: 0.6319
    Epoch 5/300
    891/891 [==============================] - 0s 35us/step - loss: 0.6407 - acc: 0.6364
    Epoch 6/300
    891/891 [==============================] - 0s 35us/step - loss: 0.6288 - acc: 0.6397
    Epoch 7/300
    891/891 [==============================] - 0s 35us/step - loss: 0.6189 - acc: 0.6453
    Epoch 8/300
    891/891 [==============================] - 0s 35us/step - loss: 0.6096 - acc: 0.6611
    Epoch 9/300
    891/891 [==============================] - 0s 35us/step - loss: 0.5990 - acc: 0.6835
    Epoch 10/300
    891/891 [==============================] - 0s 35us/step - loss: 0.5899 - acc: 0.7026
    Epoch 11/300
    891/891 [==============================] - 0s 35us/step - loss: 0.5839 - acc: 0.6947
    Epoch 12/300
    891/891 [==============================] - 0s 35us/step - loss: 0.5758 - acc: 0.7250
    Epoch 13/300
    891/891 [==============================] - 0s 35us/step - loss: 0.5705 - acc: 0.7228
    Epoch 14/300
    891/891 [==============================] - 0s 35us/step - loss: 0.5662 - acc: 0.7452
    Epoch 15/300
    891/891 [==============================] - 0s 35us/step - loss: 0.5641 - acc: 0.7329
    Epoch 16/300
    891/891 [==============================] - 0s 18us/step - loss: 0.5568 - acc: 0.7755
    Epoch 17/300
    891/891 [==============================] - 0s 35us/step - loss: 0.5533 - acc: 0.7699
    Epoch 18/300
    891/891 [==============================] - 0s 53us/step - loss: 0.5519 - acc: 0.7587
    Epoch 19/300
    891/891 [==============================] - ETA: 0s - loss: 0.4718 - acc: 0.800 - 0s 18us/step - loss: 0.5473 - acc: 0.7576
    Epoch 20/300
    891/891 [==============================] - 0s 18us/step - loss: 0.5436 - acc: 0.7699
    Epoch 21/300
    891/891 [==============================] - 0s 35us/step - loss: 0.5385 - acc: 0.7856
    Epoch 22/300
    891/891 [==============================] - 0s 35us/step - loss: 0.5354 - acc: 0.7789
    Epoch 23/300
    891/891 [==============================] - 0s 53us/step - loss: 0.5325 - acc: 0.7811
    Epoch 24/300
    891/891 [==============================] - 0s 35us/step - loss: 0.5298 - acc: 0.7800
    Epoch 25/300
    891/891 [==============================] - 0s 35us/step - loss: 0.5279 - acc: 0.7890
    Epoch 26/300
    891/891 [==============================] - 0s 18us/step - loss: 0.5259 - acc: 0.7834
    Epoch 27/300
    891/891 [==============================] - 0s 35us/step - loss: 0.5229 - acc: 0.7868
    Epoch 28/300
    891/891 [==============================] - 0s 53us/step - loss: 0.5200 - acc: 0.7879
    Epoch 29/300
    891/891 [==============================] - 0s 18us/step - loss: 0.5147 - acc: 0.7823
    Epoch 30/300
    891/891 [==============================] - 0s 35us/step - loss: 0.5177 - acc: 0.7969
    Epoch 31/300
    891/891 [==============================] - 0s 35us/step - loss: 0.5105 - acc: 0.7969
    Epoch 32/300
    891/891 [==============================] - 0s 35us/step - loss: 0.5076 - acc: 0.7957
    Epoch 33/300
    891/891 [==============================] - 0s 35us/step - loss: 0.5052 - acc: 0.7924
    Epoch 34/300
    891/891 [==============================] - 0s 35us/step - loss: 0.5058 - acc: 0.8070
    Epoch 35/300
    891/891 [==============================] - 0s 35us/step - loss: 0.5061 - acc: 0.7991
    Epoch 36/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4999 - acc: 0.8013
    Epoch 37/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4984 - acc: 0.8025
    Epoch 38/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4961 - acc: 0.8047
    Epoch 39/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4957 - acc: 0.8070
    Epoch 40/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4933 - acc: 0.7991
    Epoch 41/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4892 - acc: 0.8070
    Epoch 42/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4898 - acc: 0.8036
    Epoch 43/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4894 - acc: 0.8036
    Epoch 44/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4864 - acc: 0.8092
    Epoch 45/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4838 - acc: 0.8081
    Epoch 46/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4820 - acc: 0.8204
    Epoch 47/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4837 - acc: 0.8092
    Epoch 48/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4796 - acc: 0.8070
    Epoch 49/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4840 - acc: 0.8148
    Epoch 50/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4774 - acc: 0.8137
    Epoch 51/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4756 - acc: 0.8103
    Epoch 52/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4761 - acc: 0.8137
    Epoch 53/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4729 - acc: 0.8204
    Epoch 54/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4737 - acc: 0.8182
    Epoch 55/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4715 - acc: 0.8193
    Epoch 56/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4730 - acc: 0.8182
    Epoch 57/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4734 - acc: 0.8126
    Epoch 58/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4732 - acc: 0.8126
    Epoch 59/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4693 - acc: 0.8171
    Epoch 60/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4691 - acc: 0.8126
    Epoch 61/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4672 - acc: 0.8148
    Epoch 62/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4659 - acc: 0.8148
    Epoch 63/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4672 - acc: 0.8081
    Epoch 64/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4700 - acc: 0.8081
    Epoch 65/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4658 - acc: 0.8148
    Epoch 66/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4628 - acc: 0.8159
    Epoch 67/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4616 - acc: 0.8159
    Epoch 68/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4617 - acc: 0.8204
    Epoch 69/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4600 - acc: 0.8227
    Epoch 70/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4609 - acc: 0.8159
    Epoch 71/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4596 - acc: 0.8215
    Epoch 72/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4572 - acc: 0.8238
    Epoch 73/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4603 - acc: 0.8171
    Epoch 74/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4572 - acc: 0.8182
    Epoch 75/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4571 - acc: 0.8171
    Epoch 76/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4565 - acc: 0.8227
    Epoch 77/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4552 - acc: 0.8182
    Epoch 78/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4551 - acc: 0.8193
    Epoch 79/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4583 - acc: 0.8215
    Epoch 80/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4577 - acc: 0.8204
    Epoch 81/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4555 - acc: 0.8137
    Epoch 82/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4552 - acc: 0.8114
    Epoch 83/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4530 - acc: 0.8193
    Epoch 84/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4519 - acc: 0.8215
    Epoch 85/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4545 - acc: 0.8193
    Epoch 86/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4536 - acc: 0.8193
    Epoch 87/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4521 - acc: 0.8193
    Epoch 88/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4517 - acc: 0.8204
    Epoch 89/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4477 - acc: 0.8193
    Epoch 90/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4520 - acc: 0.8193
    Epoch 91/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4483 - acc: 0.8193
    Epoch 92/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4484 - acc: 0.8182
    Epoch 93/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4502 - acc: 0.8227
    Epoch 94/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4557 - acc: 0.8148
    Epoch 95/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4523 - acc: 0.8171
    Epoch 96/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4478 - acc: 0.8238
    Epoch 97/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4454 - acc: 0.8159
    Epoch 98/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4449 - acc: 0.8204
    Epoch 99/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4462 - acc: 0.8215
    Epoch 100/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4465 - acc: 0.8215
    Epoch 101/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4455 - acc: 0.8171
    Epoch 102/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4462 - acc: 0.8171
    Epoch 103/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4426 - acc: 0.8204
    Epoch 104/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4439 - acc: 0.8260
    Epoch 105/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4432 - acc: 0.8193
    Epoch 106/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4502 - acc: 0.8215
    Epoch 107/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4444 - acc: 0.8171
    Epoch 108/300
    891/891 [==============================] - 0s 53us/step - loss: 0.4441 - acc: 0.8215
    Epoch 109/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4435 - acc: 0.8159
    Epoch 110/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4486 - acc: 0.8148
    Epoch 111/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4457 - acc: 0.8171
    Epoch 112/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4425 - acc: 0.8227
    Epoch 113/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4439 - acc: 0.8193
    Epoch 114/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4413 - acc: 0.8204
    Epoch 115/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4410 - acc: 0.8171
    Epoch 116/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4404 - acc: 0.8182
    Epoch 117/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4415 - acc: 0.8159
    Epoch 118/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4431 - acc: 0.8148
    Epoch 119/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4395 - acc: 0.8260
    Epoch 120/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4399 - acc: 0.8227
    Epoch 121/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4407 - acc: 0.8204
    Epoch 122/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4389 - acc: 0.8204
    Epoch 123/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4372 - acc: 0.8204
    Epoch 124/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4395 - acc: 0.8193
    Epoch 125/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4394 - acc: 0.8204
    Epoch 126/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4373 - acc: 0.8227
    Epoch 127/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4379 - acc: 0.8215
    Epoch 128/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4418 - acc: 0.8182
    Epoch 129/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4381 - acc: 0.8204
    Epoch 130/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4353 - acc: 0.8171
    Epoch 131/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4371 - acc: 0.8193
    Epoch 132/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4412 - acc: 0.8148
    Epoch 133/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4376 - acc: 0.8159
    Epoch 134/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4357 - acc: 0.8238
    Epoch 135/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4351 - acc: 0.8227
    Epoch 136/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4359 - acc: 0.8193
    Epoch 137/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4355 - acc: 0.8204
    Epoch 138/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4349 - acc: 0.8171
    Epoch 139/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4364 - acc: 0.8204
    Epoch 140/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4367 - acc: 0.8193
    Epoch 141/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4340 - acc: 0.8204
    Epoch 142/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4334 - acc: 0.8249
    Epoch 143/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4339 - acc: 0.8171
    Epoch 144/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4349 - acc: 0.8204
    Epoch 145/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4326 - acc: 0.8238
    Epoch 146/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4322 - acc: 0.8204
    Epoch 147/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4314 - acc: 0.8193
    Epoch 148/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4342 - acc: 0.8182
    Epoch 149/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4311 - acc: 0.8171
    Epoch 150/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4363 - acc: 0.8227
    Epoch 151/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4348 - acc: 0.8159
    Epoch 152/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4334 - acc: 0.8182
    Epoch 153/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4284 - acc: 0.8238
    Epoch 154/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4322 - acc: 0.8193
    Epoch 155/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4307 - acc: 0.8215
    Epoch 156/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4311 - acc: 0.8227
    Epoch 157/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4306 - acc: 0.8193
    Epoch 158/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4319 - acc: 0.8215
    Epoch 159/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4272 - acc: 0.8249
    Epoch 160/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4288 - acc: 0.8238
    Epoch 161/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4277 - acc: 0.8204
    Epoch 162/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4277 - acc: 0.8260
    Epoch 163/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4309 - acc: 0.8159
    Epoch 164/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4294 - acc: 0.8238
    Epoch 165/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4290 - acc: 0.8204
    Epoch 166/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4262 - acc: 0.8215
    Epoch 167/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4261 - acc: 0.8249
    Epoch 168/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4267 - acc: 0.8238
    Epoch 169/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4264 - acc: 0.8227
    Epoch 170/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4278 - acc: 0.8238
    Epoch 171/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4260 - acc: 0.8238
    Epoch 172/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4273 - acc: 0.8238
    Epoch 173/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4257 - acc: 0.8249
    Epoch 174/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4240 - acc: 0.8238
    Epoch 175/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4247 - acc: 0.8283
    Epoch 176/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4245 - acc: 0.8272
    Epoch 177/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4290 - acc: 0.8272
    Epoch 178/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4241 - acc: 0.8283
    Epoch 179/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4284 - acc: 0.8249
    Epoch 180/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4245 - acc: 0.8260
    Epoch 181/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4271 - acc: 0.8272
    Epoch 182/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4275 - acc: 0.8249
    Epoch 183/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4242 - acc: 0.8283
    Epoch 184/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4247 - acc: 0.8283
    Epoch 185/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4227 - acc: 0.8272
    Epoch 186/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4220 - acc: 0.8283
    Epoch 187/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4257 - acc: 0.8260
    Epoch 188/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4236 - acc: 0.8272
    Epoch 189/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4265 - acc: 0.8294
    Epoch 190/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4253 - acc: 0.8305
    Epoch 191/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4232 - acc: 0.8249
    Epoch 192/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4210 - acc: 0.8283
    Epoch 193/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4220 - acc: 0.8260
    Epoch 194/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4240 - acc: 0.8238
    Epoch 195/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4204 - acc: 0.8260
    Epoch 196/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4210 - acc: 0.8260
    Epoch 197/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4201 - acc: 0.8305
    Epoch 198/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4212 - acc: 0.8305
    Epoch 199/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4210 - acc: 0.8272
    Epoch 200/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4211 - acc: 0.8283
    Epoch 201/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4201 - acc: 0.8260
    Epoch 202/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4185 - acc: 0.8305
    Epoch 203/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4209 - acc: 0.8283
    Epoch 204/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4224 - acc: 0.8305
    Epoch 205/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4206 - acc: 0.8260
    Epoch 206/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4230 - acc: 0.8204
    Epoch 207/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4190 - acc: 0.8283
    Epoch 208/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4248 - acc: 0.8294
    Epoch 209/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4209 - acc: 0.8283
    Epoch 210/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4213 - acc: 0.8283
    Epoch 211/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4203 - acc: 0.8283
    Epoch 212/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4193 - acc: 0.8294
    Epoch 213/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4190 - acc: 0.8283
    Epoch 214/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4212 - acc: 0.8294
    Epoch 215/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4217 - acc: 0.8272
    Epoch 216/300
    891/891 [==============================] - ETA: 0s - loss: 0.4674 - acc: 0.800 - 0s 35us/step - loss: 0.4182 - acc: 0.8305
    Epoch 217/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4182 - acc: 0.8272
    Epoch 218/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4203 - acc: 0.8249
    Epoch 219/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4210 - acc: 0.8272
    Epoch 220/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4180 - acc: 0.8283
    Epoch 221/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4179 - acc: 0.8283
    Epoch 222/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4160 - acc: 0.8283
    Epoch 223/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4184 - acc: 0.8283
    Epoch 224/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4182 - acc: 0.8294
    Epoch 225/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4184 - acc: 0.8260
    Epoch 226/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4197 - acc: 0.8283
    Epoch 227/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4188 - acc: 0.8283
    Epoch 228/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4173 - acc: 0.8272
    Epoch 229/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4168 - acc: 0.8294
    Epoch 230/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4185 - acc: 0.8294
    Epoch 231/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4174 - acc: 0.8283
    Epoch 232/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4176 - acc: 0.8272
    Epoch 233/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4171 - acc: 0.8294
    Epoch 234/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4165 - acc: 0.8294
    Epoch 235/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4165 - acc: 0.8305
    Epoch 236/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4168 - acc: 0.8260
    Epoch 237/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4160 - acc: 0.8283
    Epoch 238/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4173 - acc: 0.8305
    Epoch 239/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4166 - acc: 0.8260
    Epoch 240/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4177 - acc: 0.8272
    Epoch 241/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4175 - acc: 0.8305
    Epoch 242/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4175 - acc: 0.8294
    Epoch 243/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4203 - acc: 0.8283
    Epoch 244/300
    891/891 [==============================] - 0s 53us/step - loss: 0.4218 - acc: 0.8316
    Epoch 245/300
    891/891 [==============================] - 0s 53us/step - loss: 0.4258 - acc: 0.8249
    Epoch 246/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4163 - acc: 0.8294
    Epoch 247/300
    891/891 [==============================] - 0s 36us/step - loss: 0.4195 - acc: 0.8260
    Epoch 248/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4233 - acc: 0.8227
    Epoch 249/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4199 - acc: 0.8283
    Epoch 250/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4190 - acc: 0.8272
    Epoch 251/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4177 - acc: 0.8283
    Epoch 252/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4213 - acc: 0.8339
    Epoch 253/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4220 - acc: 0.8294
    Epoch 254/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4182 - acc: 0.8305
    Epoch 255/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4150 - acc: 0.8294
    Epoch 256/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4167 - acc: 0.8260
    Epoch 257/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4161 - acc: 0.8283
    Epoch 258/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4162 - acc: 0.8283
    Epoch 259/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4144 - acc: 0.8283
    Epoch 260/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4167 - acc: 0.8305
    Epoch 261/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4154 - acc: 0.8294
    Epoch 262/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4186 - acc: 0.8260
    Epoch 263/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4179 - acc: 0.8260
    Epoch 264/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4168 - acc: 0.8272
    Epoch 265/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4141 - acc: 0.8294
    Epoch 266/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4243 - acc: 0.8283
    Epoch 267/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4166 - acc: 0.8260
    Epoch 268/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4164 - acc: 0.8260
    Epoch 269/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4164 - acc: 0.8294
    Epoch 270/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4167 - acc: 0.8305
    Epoch 271/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4149 - acc: 0.8283
    Epoch 272/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4154 - acc: 0.8272
    Epoch 273/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4187 - acc: 0.8260
    Epoch 274/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4159 - acc: 0.8260
    Epoch 275/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4139 - acc: 0.8294
    Epoch 276/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4152 - acc: 0.8249
    Epoch 277/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4159 - acc: 0.8260
    Epoch 278/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4188 - acc: 0.8238
    Epoch 279/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4202 - acc: 0.8272
    Epoch 280/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4174 - acc: 0.8305
    Epoch 281/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4154 - acc: 0.8238
    Epoch 282/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4209 - acc: 0.8249
    Epoch 283/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4181 - acc: 0.8294
    Epoch 284/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4153 - acc: 0.8272
    Epoch 285/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4136 - acc: 0.8305
    Epoch 286/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4175 - acc: 0.8272
    Epoch 287/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4150 - acc: 0.8283
    Epoch 288/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4154 - acc: 0.8249
    Epoch 289/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4138 - acc: 0.8283
    Epoch 290/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4139 - acc: 0.8294
    Epoch 291/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4150 - acc: 0.8272
    Epoch 292/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4148 - acc: 0.8305
    Epoch 293/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4151 - acc: 0.8272
    Epoch 294/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4135 - acc: 0.8272
    Epoch 295/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4152 - acc: 0.8283
    Epoch 296/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4171 - acc: 0.8283
    Epoch 297/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4164 - acc: 0.8215
    Epoch 298/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4199 - acc: 0.8294
    Epoch 299/300
    891/891 [==============================] - 0s 35us/step - loss: 0.4152 - acc: 0.8272
    Epoch 300/300
    891/891 [==============================] - 0s 18us/step - loss: 0.4145 - acc: 0.8305
    




    <keras.callbacks.History at 0x198f5d8d5f8>




```python
y_pred_nn = np.round(model.predict(X_test))
y_pred_nn_list = []
for i in range(len(y_pred_nn)):
    y_pred_nn_list.append(int(y_pred_nn[i][0]))

```


```python
nn_etc = accuracy_score(y_pred_nn,y_test)
print("nn acc: ",nn_etc)
predictions.to_csv("predictions.csv",index='PassengerId')
```

    nn acc:  0.9641148325358851
    
