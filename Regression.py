from Preprocess import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

i = 0
j = 0
k = 0
a = ['MinMaxScalar', 'Normalizer', 'RobustScalar', 'StandardScalar']
print('_________________________________________________________')
print('_________________________________________________________')
print('THE VALUES FOR LINEAR REGRESSION ARE:')
while i < 4:
    print('For', a[i], ' the value of accuracy is:')
    lr = LinearRegression().fit(X_trainval_[i], y_trainval)
    print("Test score: {:.2f}".format(lr.score(X_test_[i], y_test)))
    print('_________________________________________________________')
    i = i + 1

print('_________________________________________________________')
print('_________________________________________________________')
print('THE VALUES FOR RIDGE REGRESSION ARE:')
while j < 4:
    print('For', a[j], ' the value of accuracy is:')
    r = Ridge().fit(X_trainval_[j], y_trainval)
    print("Test score: {:.2f}".format(r.score(X_test_[j], y_test)))
    print('_________________________________________________________')
    j = j + 1


