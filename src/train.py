import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

data_dict = pickle.load(open('artifacts/data.pickle','rb'))
print(len(data_dict['labels']))
print(len(data_dict['data']))

data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.2,shuffle=True,stratify=labels)

model = RandomForestClassifier()

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(accuracy_score(y_pred,y_test))
print(classification_report(y_test,y_pred))

with open('artifacts/model.pkl','wb') as f:
    pickle.dump({'model':model},f)