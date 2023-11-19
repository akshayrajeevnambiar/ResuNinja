import numpy as np
import pandas as pd
from sklearn.svm import SVC
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from models.dataPreprocessing.wordProcessor import WordProcessor
from models.dataPreprocessing.constants.modelConstants import ModelConstants

class Model:

    def __init__(self):

        
        resume_data_set = pd.read_csv("data/UpdatedResumeDataSet.csv")

        self.x = resume_data_set['Resume']
        self.y = resume_data_set['Category']

        # Instantiate WordProcessor
        self.wordProcessor = WordProcessor()

        # Perform train-test split
        self.x_train, self.x_test, self.y_train, self.y_test = self.wordProcessor.custom_train_test_split(
            x = self.x,
            y = self.y
        )

        

    """
       Creates a Logistic Regression model and fits it using the training data.

       Returns:
           Predicted values of the trained SVM model.
    """

    def logistic_regression(self) -> np.ndarray:
        model = LogisticRegression(
            multi_class = 'auto',
            max_iter = ModelConstants.MAX_ITERS
        )
        model.fit(self.x_train, self.y_train)

        return model.predict(self.x_test)

    """
           Creates a SVM model and fits it using the training data.

           Returns:
               Predicted values of the trained SVM model.
    """

    def svm(self) -> np.ndarray:
        model = SVC(kernel = 'linear')
        model.fit(self.x_train, self.y_train)

        return model.predict(self.x_test)

    """
            Creates a Naive Bayes model and fits it using the training data.

            Returns:
                Predicted values of the trained Naive Bayes model.
    """

    def naive_bayes(self) -> np.ndarray:
        model = MultinomialNB()
        model.fit(self.x_train, self.y_train)

        return model.predict(self.x_test)

    """
        Creates a KNN model and fits it using the training data.

        Returns:
            Predicted values of the trained KNN model.
    """

    def knn(self) -> np.ndarray:
        model = OneVsRestClassifier(KNeighborsClassifier())
        model.fit(self.x_train, self.y_train)

        return model.predict(self.x_test)

    """
        Creates a Random Forest Regressor model and fits it using the training data.

        Returns:
            Predicted values of the trained Random Forest Regressor model.
    """

    def random_forest_regressor(self) -> np.ndarray:
        model = RandomForestClassifier(
            n_estimators = ModelConstants.NUMBER_OF_ESTIMATORS,
            random_state = ModelConstants.RANDOM_STATE
        )
        model.fit(self.x_train, self.y_train)

        return model.predict(self.x_test)

    @staticmethod
    def get_classification_report(test, predicted) -> tuple:
        accuracy = accuracy_score(test, predicted)
        classification_rep = classification_report(test, predicted)

        return accuracy, classification_rep

    @staticmethod
    def load_train_models():
        # Example usage
        # Generate synthetic data for demonstration
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
        X_sparse = csr_matrix(X)
        # Instantiate the Models class
        models_instance = Models(X_sparse, y)

        # Train and evaluate Logistic Regression
        logistic_regression_predictions = models_instance.logistic_regression()
        accuracy_lr, report_lr = models_instance.get_classification_report(models_instance.y_test, logistic_regression_predictions)

        print("Logistic Regression Accuracy:", accuracy_lr)
        print("Logistic Regression Classification Report:\n", report_lr)

        # Train and evaluate Support Vector Machine
        svm_predictions = models_instance.svm()
        accuracy_svm, report_svm = models_instance.get_classification_report(models_instance.y_test, svm_predictions)

        print("\nSVM Accuracy:", accuracy_svm)
        print("SVM Classification Report:\n", report_svm)

        # Similarly, train and evaluate other models...

        # Train and evaluate Naive Bayes
        nb_predictions = models_instance.naive_bayes()
        accuracy_nb, report_nb = models_instance.get_classification_report(models_instance.y_test, nb_predictions)

        print("\nNaive Bayes Accuracy:", accuracy_nb)
        print("Naive Bayes Classification Report:\n", report_nb)

        return models_instance
    
    def classify_resume(self, inputResume):
        return 1
    def classify_resumes(self, inputResumes):
        return 1
    

models = Model()