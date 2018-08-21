import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer


X = pd.read_csv('Features.csv')
y = pd.read_csv('Responses.csv')
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y.values.ravel(), random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_trainval, y_trainval, random_state=0)
MMS = MinMaxScaler()
SS = StandardScaler()
RS = RobustScaler()
NS = Normalizer()

for x in [X_trainval, X_train, X_valid, X_test]:
    MMS.fit(x)
    SS.fit(x)
    NS.fit(x)
    RS.fit(x)

X_train_m = MMS.transform(X_train)
X_test_m = MMS.transform(X_test)
X_valid_m = MMS.transform(X_valid)
X_trainval_m = MMS.transform(X_trainval)
X_valid_s = SS.transform(X_valid)
X_train_s = SS.transform(X_train)
X_test_s = SS.transform(X_test)
X_trainval_s = MMS.transform(X_trainval)
X_valid_n = NS.transform(X_valid)
X_trainval_n = NS.transform(X_trainval)
X_train_n = NS.transform(X_train)
X_test_n = NS.transform(X_test)
X_train_r = RS.transform(X_train)
X_test_r = RS.transform(X_test)
X_valid_r = MMS.transform(X_valid)
X_trainval_r = RS.transform(X_trainval)
