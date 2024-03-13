# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data
2. Print the placement data and salary data.
3. Find the null and duplicate values
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```python
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Syed Mokthiyar S.M
RegisterNumber:212222230156
*/
import pandas as pd

data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data.head()

data1=data1.drop(['sl_no','salary'],axis=1)

data1.isnull().sum()

data1.duplicated().sum()

data1

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)
print("Accuracy Score:",accuracy)
print("\nConfusion Matrix:\n",confusion)
print ("\nClassification Report:\n",cr)

from sklearn import metrics
cm_display =metrics.ConfusionMatrixDisplay(confusion_matrix = confusion,display_labels=[True,False])
cm_display.plot()


```

## Output:
## TOP 5 ELEMENTS
![Screenshot 2024-03-12 162221](https://github.com/syedmokthiyar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118787294/8d3d662b-b39a-48a1-88c4-41522f1775d7)

## Data-Status:
![Screenshot 2024-03-12 205614](https://github.com/syedmokthiyar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118787294/3b09c004-0754-4e33-b7ad-f8f2af7834dc)

## y_prediction array:
![Screenshot 2024-03-12 210238](https://github.com/syedmokthiyar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118787294/80b6b895-9de4-4ac7-94ae-510aaab35d32)

## Classification Report:
![Screenshot 2024-03-12 205342](https://github.com/syedmokthiyar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118787294/64953543-3a46-4b81-8ac7-a854c0249f13)

## Graph:
![Screenshot 2024-03-12 205106](https://github.com/syedmokthiyar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118787294/7bfed7d0-01cb-423c-b714-8f6a879213fa)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
