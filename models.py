from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# function for each classifer with their optimum parameters
def logistic_regression():
    return LogisticRegression(C=10, l1_ratio=0.5, penalty='elasticnet', solver='saga', max_iter=3000, random_state=42)

def decision_tree():
    return DecisionTreeClassifier(max_depth=10, min_samples_split=2, min_samples_leaf=3)

def random_forest():
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=3, min_samples_split=2, max_depth=10, criterion="gini", random_state=42)

def knn():
    return KNeighborsClassifier(n_neighbors=11, metric='euclidean')

def gaussian_naive_bayes():
    return GaussianNB(var_smoothing=0.00001)

def svm():
    return SVC(kernel="rbf", gamma=1, C=10, class_weight=None)

def ann():
    return MLPClassifier(hidden_layer_sizes=(100,), activation='tanh', max_iter=2000, random_state=42, solver="lbfgs", alpha=0.01, learning_rate="constant", learning_rate_init=0.001)

# Model definition
lr_clf = logistic_regression()
dt_clf = decision_tree()
knn_clf = knn()
rf_clf = random_forest()
g_nb_clf = gaussian_naive_bayes()
svm_clf = svm()
ann_clf = ann()