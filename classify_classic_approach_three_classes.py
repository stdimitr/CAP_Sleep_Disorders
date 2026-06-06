from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
import scipy.io
import numpy as np
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

# create dataset
#X, y = make_classification(n_samples=1000, n_features=20, random_state=1, n_informative=10, n_redundant=10)
total_normal_subjects= 16

X = scipy.io.loadmat('features.mat')
X= X['features']

Y= scipy.io.loadmat('total_classes.mat')
Y= Y['total_classes']
Y= Y.ravel()

#X = X.astype(np.float32)

X_normal= X[:total_normal_subjects, :]
Y_normal= Y[:total_normal_subjects]

indices = np.where(np.isin(Y, [1, 5, 7]))[0]

X= X[indices, :]
Y= Y[indices]

total_folds= 5

# configure the cross-validation procedure
cv_outer = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=1)

# enumerate splits
outer_results = list()

total_accuracy= []
total_balanced_accuracy= []
total_y_real= []
total_y_pred= []
total_feature_importance= []

fold= 1

for train_ix, test_ix in cv_outer.split(X, Y):

    print('Calculate outer fold ', fold)
    fold += 1
    
    # split data
    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = Y[train_ix], Y[test_ix]

    # configure the cross-validation procedure
    cv_inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
    # define the model
    model = RandomForestClassifier(oob_score=True, max_features="sqrt", random_state=123)
    # define search space

    space = dict()
    space['n_estimators'] = [100,200,500,1000]
    space['max_features'] = [5,10,20,50,105]

    # define search
    search = GridSearchCV(model, space, scoring='balanced_accuracy', cv=cv_inner, refit=True)
    # execute search
    
    smote = SMOTE(
        random_state=42,
        k_neighbors=2
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train,
    )
    
    result = search.fit(X_train_smote, y_train_smote)
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_

    importance = best_model.feature_importances_
    total_feature_importance.append(importance)

    print("\nBest Parameters:")
    print(search.best_params_)

    print("\nBest Inner CV Score:")
    print(search.best_score_)
    
    # evaluate model on the hold out dataset
    y_pred = best_model.predict(X_test)

    total_y_pred.extend(y_pred)
    total_y_real.extend(y_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    total_accuracy.append(acc)
    total_balanced_accuracy.append(bal_acc)

    print("\nAccuracy:", acc)
    print("Balanced Accuracy:", bal_acc)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))


total_accuracy= np.array(total_accuracy)
total_balanced_accuracy= np.array(total_balanced_accuracy)

total_y_real= np.array(total_y_real)
total_y_pred= np.array(total_y_pred)

print("\nMean Accuracy:")
print(np.mean(total_accuracy))

print("\nMean Balanced Accuracy:")
print(np.mean(total_balanced_accuracy))

cm = confusion_matrix(total_y_real, total_y_pred)
print("\nConfusion Matrix:\n")
print(cm)

print("Global Classification Report")
print(classification_report(total_y_real, total_y_pred))

total_feature_importance = np.array(total_feature_importance)
mean_importance = np.mean(total_feature_importance, axis=0)
best_feature_indices = np.argsort(mean_importance)[::-1]

print("Best features:")
print(np.sort(mean_importance)[::-1])

print("Best features (ranked indices):")
print(best_feature_indices)
