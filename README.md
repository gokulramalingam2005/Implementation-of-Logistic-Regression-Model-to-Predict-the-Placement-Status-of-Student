# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array. 
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values

## Program:


Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: singaravetrivel S

RegisterNumber:  212222220048
```python 
import pandas as pd
data=pd.read_csv('/content/Placement_Data (1).csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size =0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver ="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy


from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

## 1.Placement Data
![image](https://github.com/ATHMAJ03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118753139/98104a9c-dc6d-49d1-a9a1-99069af6367f)

## 2.Salary Data
![image](https://github.com/ATHMAJ03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118753139/e73ae91f-94af-4dc6-836f-dd7661c0fb4b)

## 3. Checking the null function()
![image](https://github.com/ATHMAJ03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118753139/b94f186b-736f-4f28-8ec3-12eb847a2feb)

## 4.Data Duplicate
![image](https://github.com/ATHMAJ03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118753139/55cc1c42-99fa-4453-b092-c0d6634ec596)

## 5.Print Data
![image](https://github.com/ATHMAJ03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118753139/94ed526c-fd9f-434f-b66e-de39971382e5)

## 6.Data Status
![image](https://github.com/ATHMAJ03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118753139/bdf7f80e-6dd3-4539-9e66-b5596821af24)

## 7.y_prediction array
![image](https://github.com/ATHMAJ03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118753139/a1cb327e-1f42-4da1-83d9-b14ddf7d0e23)

## 8.Accuracy value

![image](https://github.com/ATHMAJ03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118753139/b10dee90-eaa6-44b8-a6f9-23c3e909545c)

## 9.Confusion matrix
![image](https://github.com/ATHMAJ03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118753139/d9bdc644-a49a-407f-942a-715773f0ca65)

## 10.Classification Report
![image](https://github.com/ATHMAJ03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118753139/e659e723-be3b-4ceb-9856-c421c0170031)

## 11.Prediction of LR
![image](https://github.com/ATHMAJ03/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118753139/9f33b586-a40c-40c6-8829-8006afd6a07d)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
