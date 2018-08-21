from Preprocess import *
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

knn = KNeighborsRegressor(metric='manhattan').fit(X_trainval_m, y_trainval)

clf = cross_val_score(knn, X_test_m, y_test)
print("The Test score for the dataset is", clf*100)
