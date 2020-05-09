# Import libraries
import numpy as np
print('imported numpy')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import pickle



print('imported all')

data_table = pd.read_csv('postags_lemmas_levels_data.csv')
data_table = data_table.drop(['Unnamed: 0','tekstikood', 'filename'], 1)

print('read data')

labelencoder_0 = LabelEncoder() #independent variable encoder
data_table.iloc[:,17] = labelencoder_0.fit_transform(data_table.iloc[:,17])

#Transforming values into percentages of total and splitting into target and features
features = data_table.loc[:, "A":"Z"]
target_var = data_table.loc[:, "keeletase"]

print('split to test and train')
X_train, X_test, y_train, y_test =\
    train_test_split(features.loc[:,'A':"Z"], target_var, test_size = 0.5, random_state=1111)


print('=' * 80)
print("Decision Tree")
dt_param_grid = {
    'max_depth': [5,7, None],
    'criterion': ['gini', 'entropy']
}
dt = DecisionTreeClassifier()
dt_search = GridSearchCV(dt, dt_param_grid, cv=3, return_train_score=True)
dt_search.fit(X_train, y_train)
classifier = dt_search.best_estimator_
prediction = classifier.predict(X_test)

filename = 'dtree.sav'
pickle.dump(classifier, open(filename, 'wb'))


# Save the model to disk
# joblib.dump(classifier, 'classifier.joblib')





