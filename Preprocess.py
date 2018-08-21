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
MMS = MinMaxScaler()   # Take as 0 index
NS = Normalizer()      # Take as 1 index
RS = RobustScaler()    # Take as 2 index
SS = StandardScaler()  # Take as 3 index

for x in [X_trainval, X_train, X_valid, X_test]:
    MMS.fit(x)
    SS.fit(x)
    NS.fit(x)
    RS.fit(x)
X_train_ = []
X_test_ = []
X_valid_ = []
X_trainval_ = []

X_train_.append(MMS.transform(X_train))
X_test_.append(MMS.transform(X_test))
X_valid_.append(MMS.transform(X_valid))
X_trainval_.append(MMS.transform(X_trainval))
X_valid_.append(NS.transform(X_valid))
X_trainval_.append(NS.transform(X_trainval))
X_train_.append(NS.transform(X_train))
X_test_.append(NS.transform(X_test))
X_train_.append(RS.transform(X_train))
X_test_.append(RS.transform(X_test))
X_valid_.append(RS.transform(X_valid))
X_trainval_.append(RS.transform(X_trainval))
X_valid_.append(SS.transform(X_valid))
X_train_.append(SS.transform(X_train))
X_test_.append(SS.transform(X_test))
X_trainval_.append(SS.transform(X_trainval))



