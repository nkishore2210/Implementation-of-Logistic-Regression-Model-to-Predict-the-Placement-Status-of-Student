# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: N.Kishore
RegisterNumber: 212222240049

import pandas as pd
data = pd.read_csv('dataset/Placement_Data.csv')
data.head()

data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:, :-1]
x

y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```

## Output:

![Screenshot 2023-05-14 101526](https://github.com/nkishore2210/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707090/452fe641-d7cd-42b0-bdcc-53f5500d0c7e)

![Screenshot 2023-05-14 102210](https://github.com/nkishore2210/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707090/ed52e8f8-719a-4916-97e7-8413da1bce07)

![Screenshot 2023-05-14 102248](https://github.com/nkishore2210/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707090/8252d939-b92b-408c-8c51-ee5d547589cc)

![Screenshot 2023-05-14 102325](https://github.com/nkishore2210/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707090/bb3d2934-277c-496d-89e3-cf7d35e9b2e1)

![Screenshot 2023-05-14 102353](https://github.com/nkishore2210/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707090/9dfdb7ed-3791-45e3-8aa2-b94de419e38d)

![Screenshot 2023-05-14 102425](https://github.com/nkishore2210/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707090/fb274d48-4a6c-435c-a29d-5d9661d0601d)

![Screenshot 2023-05-14 102446](https://github.com/nkishore2210/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707090/20d1b0c9-f2c3-46fe-a97d-574edd756515)

![Screenshot 2023-05-14 102507](https://github.com/nkishore2210/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707090/ed91c978-0797-4552-9ff9-d553329e92bd)

![Screenshot 2023-05-14 102523](https://github.com/nkishore2210/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707090/7903e835-b879-4b3f-8272-f65380bfcc0f)

![Screenshot 2023-05-14 102553](https://github.com/nkishore2210/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707090/7cab204a-888b-4739-8e77-ee414176dafe)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
