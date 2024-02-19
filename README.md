# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

# AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

# Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

# Program:
```python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: K Santhan 
RegisterNumber: 212222230029
*/
import pandas as pd
df=pd.read_csv('student_scores.csv')
print(df.head())
print(df.tail())
x=(df.iloc[:,:-1]).values
x
y=(df.iloc[:,1]).values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours Vs Scores(Train Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_test,regressor.predict(x_test),color="yellow")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

# Output:

## df.head()
![image](https://github.com/SANTHAN-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/80164014/a882e7d3-5812-4474-a448-068991847b19)

## df.tail()
![image](https://github.com/SANTHAN-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/80164014/1225dd4a-62e4-43d5-a9e3-4755535d549d)

## Array values of X
![image](https://github.com/SANTHAN-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/80164014/9e49c96a-62ae-4df6-8a6b-4d125a2cc826)

## Array values of Y
![image](https://github.com/SANTHAN-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/80164014/4e462c35-97a0-487c-8ace-ab075b11a28b)

## Values of Y prediction
![image](https://github.com/SANTHAN-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/80164014/5b67bd2f-97be-47ef-a9ee-cb5809007015)

## Values of Y test
![image](https://github.com/SANTHAN-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/80164014/ce5825c0-c686-48b4-956e-058ab6eafb59)

## Training set graph
![image](https://github.com/SANTHAN-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/80164014/f1f53833-dee7-4fd8-b93c-5d719990b7fe)

## Testing set graph
![image](https://github.com/SANTHAN-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/80164014/586131ba-243b-4604-9012-0cc6b7cc98c8)

## Value of MSE,MAE & RMSE
![image](https://github.com/SANTHAN-2006/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/80164014/a9df1011-9c3d-43cc-9a2c-ef298ad38c30)

# Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
