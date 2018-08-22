from sklearn.svm import SVR
from Preprocess import *

best_score = 0
i = 0
a = ['MinMaxScalar', 'Normalizer', 'RobustScalar', 'StandardScalar']
while i < 4:
    print('\nFor', a[i], ' these are the values of accuracy:')
    for g in [0.01, 0.1, 1, 10]:
        for c in [0.01, 0.1, 1, 10]:
            svm = SVR(gamma=g, C=c)
            svm.fit(X_train_[i], y_train)
            s = svm.score(X_valid_[i], y_valid)
            if s > best_score:
                best_score = s
                best_para = {'gamma': g, 'C': c}
    print("______________________________________")
    print("Best score: {:.2f}".format(best_score))
    print("Best parameters: {}".format(best_para))
    svm = SVR(**best_para)
    svm.fit(X_trainval_[i], y_trainval)
    print("Best score: {:.2f}".format(svm.score(X_test_[i], y_test)))
    print("____________________________________________________")
    i = i + 1


