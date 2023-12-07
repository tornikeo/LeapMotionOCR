from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import feature_classifier

import sys
sys.path.append('..')
import utils

X, y = utils.load_training_data("../training_data/")

# drop third coordinate if present
for i in range(len(X)):
    X[i].drop(2, axis=1, inplace=True)
    X[i] = (X[i] - X[i].mean()) / X[i].std()

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

classifier = feature_classifier.FeatureClassifier().fit(train_X, train_y)
pred = classifier.predict(test_X)
print(classification_report(test_y.to_list(), pred))
