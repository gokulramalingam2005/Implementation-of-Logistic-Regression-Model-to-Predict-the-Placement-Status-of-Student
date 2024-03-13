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
Developed by: Gokul R
RegisterNumber:212222230039
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
![Screenshot 2024-03-12 162221](https://github.com/syedmokthiyar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student./assets/118787294/3d959df2-d30e-41c1-9afa-0babc4bbda53)
## Data-Status:
![Screenshot 2024-03-12 205614](https://github.com/syedmokthiyar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student./assets/118787294/41a36001-abf9-4852-9e63-955920d5b4bd)
## y_prediction array:
![Screenshot 2024-03-12 210238](https://github.com/syedmokthiyar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student./assets/118787294/b21aa187-86eb-4db5-b256-cd8fe406a73f)

## Classification Report:
![Screenshot 2024-03-12 205342](https://github.com/syedmokthiyar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student./assets/118787294/c38411a5-02ae-47ba-a562-d65205c33ddd)
## Graph:
![Screenshot 2024-03-12 205106](https://github.com/syedmokthiyar/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student./assets/118787294/ac4b612b-1978-4a31-8941-94d50d4a2e0a)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
