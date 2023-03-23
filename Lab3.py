# Naive Bayes
# databse used: iris.csv loaded from sklearn datasets

import numpy as np
from sklearn.datasets import load_iris
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# load the iris dataset
iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target

# split data into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)

# using StandardScaler() for an easier interpretation of results
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# suppress scientific format for easier reading
np.set_printoptions(suppress=True)

# build the model
nb = GaussianNB().fit(X_train_std, y_train)

# print the probabilities and predictions of the model
print(f'Classes: {nb.classes_}')
print(f'Probability for every instance: {nb.predict_proba(X_test_std)*100}')

y_pred = nb.predict(X_test_std)

print(f'Classes for every test instance:\n {y_test}')
print(f'Predictions for every test instance:\n {y_pred}')

# calculate prediction error as: incorrect prediction / total value
n_test = X_test.shape[0]
print(f'Number of new instances: {n_test}')

n_pred_incorrect = (y_test != y_pred).sum()

print('Prediction error: % 5.2f' % (n_pred_incorrect / n_test))
pred = nb.predict(X_test_std)

print('Accuracy score: % 5.2f' % accuracy_score(y_test, pred))

# graphic reprezentation of results
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
      # set marker generator and color map
      markers = ('s', 'x', 'o', '^', 'v')
      colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
      cmap = ListedColormap(colors[:len(np.unique(y))])

      # plot decision surface
      x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
      x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
      xx1, xx2 = xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
      Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
      Z = Z.reshape(xx1.shape)
      plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
      plt.xlim(xx1.min(), xx1.max())
      plt.ylim(xx2.min(), xx2.max())

      for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],  alpha=0.8, c=colors[idx],marker=markers[idx], label=cl,edgecolor=['black'])

      #highlight test samples
      if test_idx:
            X_test, y_test = X[test_idx, :], y[test_idx]
            plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor=['black'], alpha=1.0, linewidth=1, marker='o',s=100, label='test set')

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=nb, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()