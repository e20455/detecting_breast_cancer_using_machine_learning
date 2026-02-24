import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import pickle

def evaluate_model(model, X_train, X_test, y_train, y_test, kf, model_name):
    print("-----Evaluation-----")
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
    print(f"{model_name}:")
    print(f"Cross-validation Scores: {scores}")
    print(f"Mean Cross-validation Accuracy: {scores.mean():.4f}")

    model.fit(X_train, y_train)
    
    # to only be used for most accurate model (LR) to save the model
    """
    with open('lr_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Testing Accuracy: {accuracy}")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.show()

    # classification report
    report = classification_report(y_test, y_pred, target_names=['0', '1'])
    print(report)