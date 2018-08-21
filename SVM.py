from sklearn.svm import SVR
from Preprocess import *

best_score = 0


for g in [0.01, 0.1, 1, 10]:
    for c in [0.01, 0.1, 1, 10]:
        svm = SVR(gamma=g, C=c)
        svm.fit(X_train_r, y_train)
        s = svm.score(X_valid_r, y_valid)
        if s > best_score:
            best_score = s
            best_para = {'gamma': g, 'C': c}

print("Best score: {:.2f}".format(best_score))
print("Best parameters: {}".format(best_para))

svm = SVR(**best_para)
svm.fit(X_trainval_r, y_trainval)
print("Best score: {:.2f}".format(svm.score(X_test_r, y_test)))

