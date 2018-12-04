import pandas as pd
import math
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

#import data
df = pd.read_csv('data.csv')

x=df.drop(['Height'],axis=1)
y=df.drop(['Weight'],axis=1)

#split data
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.25)

#least squares model
lr_model = linear_model.LinearRegression()
lr_model.fit(x_train, y_train)

print('LINEAR MODEL')
print('y = ' , float(lr_model.coef_[0]) , ' * x + ' , float(lr_model.intercept_))
print('Training score: {}'.format(lr_model.score(x_train, y_train)))
print('Test score: {}'.format(lr_model.score(x_test, y_test)))

y_pred = lr_model.predict(x_test)

#least squares error
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
print('RMSE: {}'.format(rmse))
print('')


#ridge model
ridge = linear_model.Ridge(alpha=1.0)
ridge.fit(x_train, y_train)

print('RIDGE MODEL')
print('y = ' , float(ridge.coef_[0]) , ' * x + ' , float(ridge.intercept_))
print('Training Score: {}'.format(ridge.score(x_train, y_train)))
print('Test Score: {}'.format(ridge.score(x_test, y_test)))
print('')

y_ridge_pred = ridge.predict(x_test)

#lasso model
lasso = linear_model.Lasso(alpha=0.01)
lasso.fit(x_train, y_train)

print('LASSO MODEL')
print('y = ' , float(lasso.coef_[0]) , ' * x + ' , float(lasso.intercept_))
print('Training Score: {}'.format(lasso.score(x_train, y_train)))
print('Test Score: {}'.format(lasso.score(x_test, y_test)))

y_lasso_pred = lasso.predict(x_test)

#plots
plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred)
plt.plot(x_test,y_ridge_pred,color='black')
plt.plot(x_test, y_lasso_pred, color='red')
plt.show()
