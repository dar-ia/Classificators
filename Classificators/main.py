import sklearn
from sklearn import datasets as ds
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB as naive_bayes
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.svm import SVC as svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

data_set = ds.load_digits()

x = data_set.data
y = data_set.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

neighbors_model = KNeighborsClassifier(n_neighbors=3)
bayes_model = naive_bayes()
tree_model = tree()
svm_model = svm()
forest_model = RandomForestClassifier()

neighbors_model.fit(X_train, y_train)
bayes_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)

y_actual_neighbors = neighbors_model.predict(X_test)
y_actual_bayes = bayes_model.predict(X_test)
y_actual_tree = tree_model.predict(X_test)
y_actual_svm = svm_model.predict(X_test)
y_actual_forest = forest_model.predict(X_test)

neighbors_metrics = metrics.classification_report(y_test, y_actual_neighbors)
bayes_metrics = metrics.classification_report(y_test, y_actual_bayes)
tree_metrics = metrics.classification_report(y_test, y_actual_tree)
svm_metrics = metrics.classification_report(y_test, y_actual_svm)
forest_metrics = metrics.classification_report(y_test, y_actual_forest)

print("KNN metrics:\n"+neighbors_metrics+"\nBayes report:\n"+bayes_metrics+"\nTree report:\n"+tree_metrics+"\nSVM metrics:\n"+svm_metrics+"Random forest metrics:\n"+forest_metrics)
