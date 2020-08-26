import sklearn
from sklearn import datasets as ds
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB as naive_bayes
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.svm import SVC as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

data_set = ds.load_digits()

x = data_set.data
y = data_set.target

cv_kfold = KFold(n_splits=30)

neighbors_classifiers = []
bayes_classifiers = []
tree_classifiers = []
svm_classifiers = []
forest_classifiers = []


for train_index, test_index in cv_kfold.split(y):
    neighbors_model = KNeighborsClassifier(n_neighbors=3)
    bayes_model = naive_bayes()
    tree_model = tree()
    svm_model = svm()
    forest_model = RandomForestClassifier()
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    neighbors_model.fit(X_train, y_train)
    bayes_model.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)
    forest_model.fit(X_train, y_train)

    neighbors_classifiers.append(neighbors_model)
    bayes_classifiers.append(bayes_model)
    tree_classifiers.append(tree_model)
    svm_classifiers.append(svm_model)

cross_neighbors = cross_val_score(KNeighborsClassifier(n_neighbors=3), x, y, cv=30)
cross_bayes = cross_val_score(naive_bayes(), x, y, cv=30)
cross_tree = cross_val_score(tree(), x, y, cv=30)
cross_svm = cross_val_score(svm(), x, y, cv=30)
cross_forest = cross_val_score(RandomForestClassifier(), x, y, cv=30)

medium_neighbors = round(sum(cross_neighbors)/len(cross_neighbors), 4)
medium_bayes = round(sum(cross_bayes)/len(cross_bayes), 4)
medium_tree = round(sum(cross_tree)/len(cross_tree), 4)
medium_svm = round(sum(cross_svm)/len(cross_svm), 4)
medium_forest = round(sum(cross_forest)/len(cross_forest), 4)

dispersion_neighbors = round(sum((medium_neighbors - x)**2 for x in cross_neighbors)/len(cross_neighbors), 4)
dispersion_bayes = round(sum((medium_bayes - x)**2 for x in cross_bayes)/len(cross_bayes), 4)
dispersion_tree = round(sum((medium_tree - x)**2 for x in cross_tree)/len(cross_tree), 4)
dispersion_svm = round(sum((medium_svm - x)**2 for x in cross_svm)/len(cross_svm), 4)
dispersion_forest = round(sum((medium_forest - x)**2 for x in cross_forest)/len(cross_forest), 4)

print("KNN: ", medium_neighbors, "+-", dispersion_neighbors)
print("Naive Bayes: ", medium_bayes, "+-", dispersion_bayes)
print("Tree: ", medium_tree, "+-", dispersion_tree)
print("SVM: ", medium_svm, "+-", dispersion_svm)
print("Random Forest: ", medium_forest, "+-", dispersion_forest)

print(cross_neighbors)



