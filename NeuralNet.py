from sklearn.neural_network import MLPRegressor
from Preprocess import *

i = 0
best_score = 0
a = ['MinMaxScalar', 'Normalizer', 'RobustScalar', 'StandardScalar']
while i < 4:
    print('For', a[i], 'the value of accuracy is:')
    for j in range(2,  100):
        nn = MLPRegressor(hidden_layer_sizes=j, activation='relu', solver='sgd', max_iter=600)
        nn.fit(X_train_[i], y_train)
        s = nn.score(X_valid_[i], y_valid)
        if s > best_score:
            best_score = s
            best_hidden_layer_size = j
    print("_________________________________________________________")
    print("Best score: {:.2f}".format(best_score))
    print("Best hidden layer size: {}".format(best_hidden_layer_size))
    nn = MLPRegressor(hidden_layer_sizes=best_hidden_layer_size, activation='logistic', max_iter=400)
    nn.fit(X_trainval_[i], y_trainval)
    print("Best score: {:.2f}".format(nn.score(X_test_[i], y_test)))
    print("____________________________________________________")
    i = i + 1
