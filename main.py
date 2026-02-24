import numpy as np
from sklearn.model_selection import KFold, train_test_split
import data_preprocessing
from evaluation import evaluate_model
from models import dt_clf, lr_clf, knn_clf, rf_clf, g_nb_clf, svm_clf, ann_clf
from optimisation import dt_grid_search, lr_grid_search, knn_grid_search, rf_grid_search, g_nb_grid_search, svm_grid_search, ann_grid_search

X = data_preprocessing.rescaled_df
y = np.array(data_preprocessing.diagnosis_outcome)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# use main.py to evaulate and train models 
# uncomment the functions, you would like to run

# train and evaulate models 
#evaluate_model(dt_clf, X_train, X_test, y_train, y_test, kf, "Decision Tree")
evaluate_model(lr_clf, X_train, X_test, y_train, y_test, kf, "Logistic Regression")
#evaluate_model(knn_clf, X_train, X_test, y_train, y_test, kf, "K-Nearest Neighbour")
#evaluate_model(rf_clf, X_train, X_test, y_train, y_test, kf, "Random Forest")
#evaluate_model(g_nb_clf, X_train, X_test, y_train, y_test, kf, "Gaussian Naive Bayes")
#evaluate_model(svm_clf, X_train, X_test, y_train, y_test, kf, "SVM")
#evaluate_model(ann_clf, X_train, X_test, y_train, y_test, kf, "ANN")

# optimise models using grid search 
#dt_grid_search(X_train, X_test, y_train, y_test, kf)
lr_grid_search(X_train, X_test, y_train, y_test, kf)
#knn_grid_search(X_train, X_test, y_train, y_test, kf)
#rf_grid_search(X_train, X_test, y_train, y_test, kf)
#g_nb_grid_search(X_train, X_test, y_train, y_test, kf)
#svm_grid_search(X_train, X_test, y_train, y_test, kf)
#ann_grid_search(X_train, X_test, y_train, y_test, kf)


