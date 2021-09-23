# Importing the necessary libraries

import pickle
from sklearn.linear_model \
import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the iris dataset from sklearn datasets
df = load_iris()

# features
names = df.feature_names

# features and labels from the dataset
features = df.data
labels = df.target

# visulizagions
plt.figure()
sns.pairplot(data=df.data, hue='Species')
plt.show()

# splitting labels and features to training and testing sets
feature_train, feature_test, label_train, label_test = train_test_split(features, labels, test_size=0.2,
                                                                        random_state=99)

# Logistic Regression Model with maximum iterations as 300
model = LogisticRegression(max_iter=300)
model.fit(feature_train, label_train)
label_pred = model.predict(feature_test)
#accuracy score for predicted ones vs testing ones
accuracy_score(label_pred, label_test)



# KNN model
knnModel = neighbors.KNeighborsClassifier(n_neighbors=5)
knnModel.fit(feature_train, label_train)
#print(knnModel)
label_pred_knn = knnModel.predict(feature_test)
print(f'\naccuracy score: {accuracy_score(label_test, label_pred_knn):.4f}')
# Cross-validation.
knn_score = cross_val_score(knnModel, feature_train, label_train, cv=10)
print(f'\nCross-Validation Scores: {knn_score}')
print(f'\nAveraged Cross-Validation Scores: {knn_score.mean()}')


# dumping the model object to save it as model.pkl file
pickle.dump(model, open('model.pkl', 'wb+'))
pickle.dump(knnModel, open('model.pkl', 'wb+'))