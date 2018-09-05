from sklearn.ensemble import RandomForestRegressor
from Preprocess import *

i = 0
best_score = 0
a = ['MinMaxScalar', 'Normalizer', 'RobustScalar', 'StandardScalar']
while i < 4:
    for n in range(2, 100):
        forests = RandomForestRegressor(n_estimators=n).fit(X_train_[i], y_train)
        s = forests.score(X_valid_[i], y_valid)
        if s > best_score:
            best_score = s
            n_depth = n
    print("For", a[i],  "Best Forest Size: {}".format(n_depth))
    f = RandomForestRegressor(n_estimators=n_depth)
    f.fit(X_trainval_[i], y_trainval)
    print("Best score: {:.2f}".format(f.score(X_test_[i], y_test)))
    print("____________________________________________________")
    i = i + 1