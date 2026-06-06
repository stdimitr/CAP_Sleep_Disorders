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

def calculate_z_score(X, X_healthy, Y):

    mean_total_features = np.empty((0, np.shape(X)[1]))
    std_total_features = np.empty((0,  np.shape(X)[1]))

    final_x= np.empty((0, np.shape(X)[1]))
    X_norm = X.copy()

    for disorder_class in np.unique(Y):
        temp_class_indices= np.where(Y == disorder_class)[0]

        temp_X= X[temp_class_indices, :]

        temp_X_healthy_disorder = np.concatenate((X_healthy, temp_X), axis=0)

        temp_mean= np.mean(temp_X_healthy_disorder, 0)
        #temp_std= np.sqrt(np.std(temp_X, 0)**2 + np.std(X_healthy, 0)**2)
        temp_std= np.std(temp_X_healthy_disorder, 0)
    
        mean_total_features= np.vstack((mean_total_features, temp_mean))
        std_total_features= np.vstack((std_total_features, temp_std))

        X_norm[temp_class_indices, :] = (X[temp_class_indices, :] - temp_mean)/temp_std

    return mean_total_features, std_total_features, X_norm

def calculate_z_score_aggregated(X):
    feat = [
        np.arange(11),
        np.arange(11, 42),
        np.arange(42, 49),
        np.arange(49, 74),
        np.arange(74, 95),
        np.arange(95, 102),
        np.arange(102, 105)
    ]

    return np.column_stack([X[:, idx].sum(axis=1) for idx in feat])
    
# create dataset
#X, y = make_classification(n_samples=1000, n_features=20, random_state=1, n_informative=10, n_redundant=10)
total_normal_subjects= 16

X = scipy.io.loadmat('features.mat')
X= X['features']

Y= scipy.io.loadmat('total_classes.mat')
Y= Y['total_classes']
Y= Y.ravel()

total_folds= 5

#X = X.astype(np.float32)

X_normal= X[:total_normal_subjects, :]
Y_normal= Y[:total_normal_subjects]

indices = np.where(np.isin(Y, [5, 7]))[0]

X_disorder= X[indices, :]
Y_disorder= Y[indices]

print(Y_disorder)

#calculate_z_score(X, Y)

#[X_sleep_disorder_subjects_norm, mean_sleep_disorder, std_sleep_disorder]= calculate_sleis_score(X, Y, total_normal_subjects)

#print('vdvsd')
#print(mean_sleep_disorder)

total_folds= 5

# configure the cross-validation procedure
outer_cv = StratifiedKFold(n_splits=total_folds, shuffle=True, random_state=1)

# enumerate splits
outer_results = list()

total_accuracy= []
final_Y= []
final_estimated_Y= []

fold= 1

total_accuracy= []
total_balanced_accuracy= []
total_y_real= []
total_y_pred= []
total_feature_importance= []

for train_ix, test_ix in outer_cv.split(X_disorder, Y_disorder):

    print('Calculate outer ', fold)

    fold += 1
    
    # split data
    X_train, X_test = X_disorder[train_ix, :], X_disorder[test_ix, :]
    y_train, y_test = Y_disorder[train_ix], Y_disorder[test_ix]

    # configure the cross-validation procedure
    cv_inner = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
    # define the model
    model = RandomForestClassifier(oob_score=True, class_weight='balanced_subsample', max_features="sqrt", random_state=123)
    # define search space

    space = dict()
    space['n_estimators'] = [100,200,500,1000]
    space['max_features'] = [5,10,20,50,105]

    # define search
    search = GridSearchCV(model, space, scoring='balanced_accuracy', cv=cv_inner, refit=True)

    smote = SMOTE(
        random_state=42,
        k_neighbors=2
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train,
    )

    space = dict()
    space['n_estimators'] = [100, 500, 1000]
    #space['max_features'] = [1, 2, 3, 4]

        # execute search

    [mean_total_features, std_total_features, X_train_z_score]= calculate_z_score(X_train_smote, X_normal, y_train)

    X_train_z_score= calculate_z_score_aggregated(X_train_z_score)

    result = search.fit(X_train_z_score, y_train_smote)
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_
    # evaluate model on the hold out dataset

    temp_acc= []

    total_probability = []
    yhat= []
   
    for i in range(X_test.shape[0]):

        X_test_sample= X_test[i]

        sample_probabilities = []

        for k in range(np.shape(mean_total_features)[0]):

            temp_x= (X_test_sample - mean_total_features[k])/std_total_features[k]
            temp_x = temp_x.reshape(1, -1)

            temp_x= calculate_z_score_aggregated(temp_x)
        
            probabilities= best_model.predict_proba(temp_x)
            sample_probabilities.append(probabilities)

        sample_probabilities= np.array(sample_probabilities)
        sample_probabilities = np.squeeze(sample_probabilities)
        #print(sample_probabilities)
        
        #sample_probabilities= np.amax(sample_probabilities, axis=1)
        sample_probabilities= np.diag(sample_probabilities)
        #sample_probabilities = np.mean(sample_probabilities, axis=0)

        if np.argmax(sample_probabilities) == 0:
            yhat.append(5)
        elif np.argmax(sample_probabilities) == 1:
            yhat.append(7)
            
        # evaluate the model
        #acc = accuracy_score(y_test, yhat)
        # store the result
        #print(acc)

    importance = best_model.feature_importances_
    total_feature_importance.append(importance)

    print("\nBest Parameters:")
    print(search.best_params_)

    print("\nBest Inner CV Score:")
    print(search.best_score_)

    y_test= np.array(y_test)
    yhat= np.array(yhat)

    print("Real:", y_test)
    print("Estimated:", yhat)

    total_y_pred.extend(yhat)
    total_y_real.extend(y_test)

    acc = accuracy_score(y_test, yhat)
    bal_acc = balanced_accuracy_score(y_test, yhat)

    total_accuracy.append(acc)
    total_balanced_accuracy.append(bal_acc)

    print("\nAccuracy:", acc)
    print("Balanced Accuracy:", bal_acc)
    print("\nClassification Report:\n")
    print(classification_report(y_test, yhat))

total_accuracy= np.array(total_accuracy)
total_balanced_accuracy= np.array(total_balanced_accuracy)

total_y_real= np.array(total_y_real)
total_y_pred= np.array(total_y_pred)

print("Global Classification Report")
print(classification_report(total_y_real, total_y_pred))

print("\nMean Accuracy:")
print(np.mean(total_accuracy))

print("\nMean Balanced Accuracy:")
print(np.mean(total_balanced_accuracy))

cm = confusion_matrix(total_y_real, total_y_pred)
print("\nConfusion Matrix:\n")
print(cm)

total_feature_importance = np.array(total_feature_importance)
mean_importance = np.mean(total_feature_importance, axis=0)
best_feature_indices = np.argsort(mean_importance)[::-1]


print("Best features:")
print(np.sort(mean_importance)[::-1])

print("Best features (ranked indices):")
print(best_feature_indices)
