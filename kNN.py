from Preprocess import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

i = 0
a = ['MinMaxScalar', 'Normalizer', 'RobustScalar', 'StandardScalar']
while i < 4:
    print('\nFor', a[i], ' these are the values of accuracy:')
    print('__________________________________________________')
    for distance in ['manhattan', 'euclidean']:
        knn = KNeighborsRegressor(metric=distance).fit(X_trainval_[i], y_trainval)
        clf = cross_val_score(knn, X_test_[i], y_test)
        print('The score for ', distance, 'distance is:', clf*100)
    print('__________________________________________________')
    i = i + 1


