from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from models import dt_clf, lr_clf, knn_clf, rf_clf, g_nb_clf, svm_clf, ann_clf

# main grid search function, used for all models
def grid_search(model, param_grid, X_train, y_train, X_test, y_test, kf, model_name):
    print("-----Optimisation-----")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kf, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{model_name}:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Cross-Validation Score: {grid_search.best_score_:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

# each model with all parameters to be tested, calls main grid search function

def dt_grid_search(X_train, X_test, y_train, y_test, kf):
    param_grid = {
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 3, 4, 5],
        'min_samples_leaf': [1, 2, 3, 4]
    }
    grid_search(dt_clf, param_grid, X_train, y_train, X_test, y_test, kf, "Decision Tree")

def lr_grid_search(X_train, X_test, y_train, y_test, kf):
    param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.1, 1, 10],
        'solver': ['saga'],
        'l1_ratio': [0.1, 0.5, 0.9]
    }
    grid_search(lr_clf, param_grid, X_train, y_train, X_test, y_test, kf, "Logistic Regression")
    
def knn_grid_search(X_train, X_test, y_train, y_test, kf):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
    }
    grid_search(knn_clf, param_grid, X_train, y_train, X_test, y_test, kf, "K-Nearest Neighbour")
    
def rf_grid_search(X_train, X_test, y_train, y_test, kf):
    param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [10, 11, 12, 13, 14, 15],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 2, 3, 4],
        'criterion': ['gini', 'entropy','log_loss']
    }
    grid_search(rf_clf, param_grid, X_train, y_train, X_test, y_test, kf, "Random Forest")

def g_nb_grid_search(X_train, X_test, y_train, y_test, kf):
    param_grid = {
       'var_smoothing': [0.00001, 0.0001, 0.001]
    }
    grid_search(g_nb_clf, param_grid, X_train, y_train, X_test, y_test, kf, "Gaussian Naive Bayes")
    
def svm_grid_search(X_train, X_test, y_train, y_test, kf):
    param_grid = {
        'C': [0.1, 1, 10, 100],  
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],  
        'class_weight': [None, 'balanced']
    }
    grid_search(svm_clf, param_grid, X_train, y_train, X_test, y_test, kf, "SVM")
    
def ann_grid_search(X_train, X_test, y_train, y_test, kf):
    param_grid = {
       'solver': ['adam', 'sgd', 'lbfgs'],            
       'activation': ['relu', 'tanh', 'logistic'],
       'hidden_layer_sizes': [(100,), (50, 50), (200,)],
       'alpha': [0.0001, 0.001, 0.01, 0.1],            
       'learning_rate': ['constant', 'invscaling', 'adaptive'],  
       'learning_rate_init': [0.001, 0.01, 0.1],             
   }
    grid_search(ann_clf, param_grid, X_train, y_train, X_test, y_test, kf, "ANN")