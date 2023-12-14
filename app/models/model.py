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

from ironpdf import *
import textract


License.LicenseKey = "IRONSUITE.VISMITAPAVDIGHADA.LOYALISTCOLLEGE.COM.30484-56E50DBD95-ESG3G-5C3SV7G4PQVG-JCZUKEF5GK4D-YJIJ7GNWNCCS-Z5RVJIKJQCL7-DZUBIE3MCVM4-K4CLFBPZ7RC7-2Z3ORS-TPOY5UG3QXSLEA-DEPLOYMENT.TRIAL-RDZXHS.TRIAL.EXPIRES.26.DEC.2023"


class Model:


    def __init__(self, rawData = [], wordProcessor = WordProcessor()):

        resume_data_set = pd.read_csv("data/UpdatedResumeDataSet.csv")

        self.rawData = resume_data_set

        self.wordProcessor = WordProcessor()

        self.load_train_models(self)

        

    """
       Creates a Logistic Regression model and fits it using the training data.

       Returns:
           Predicted values of the trained SVM model.
    """

    def logistic_regression(self) -> np.ndarray:
        self.logisticRegressionModel = LogisticRegression(
            multi_class = 'auto',
            max_iter = ModelConstants.MAX_ITERS
        )
        print("----------------------")
        print(self.x_train.shape)
        print(self.y_train.shape)
        print("----------------------")

        self.logisticRegressionModel.fit(self.x_train, self.y_train)

        return self.logisticRegressionModel.predict(self.x_test)
    
    """
       Utilizes global Logistic Regression to return prediction.

       Returns:
           Predicted values for the test set
    """

    def predict_logistic_regression(self, inputResumesDf) -> np.ndarray:
        return self.logisticRegressionModel.predict(inputResumesDf)

    """
           Creates a SVM model and fits it using the training data.

           Returns:
               Predicted values of the trained SVM model.
    """

    def svm(self) -> np.ndarray:
        self.svmModel = SVC(kernel = 'linear')
        self.svmModel.fit(self.x_train, self.y_train)

        return self.svmModel.predict(self.x_test)
    
    def predict_svm(self, inputResumesDf) -> np.ndarray:
        return self.svmModel.predict(inputResumesDf)

    """
            Creates a Naive Bayes model and fits it using the training data.

            Returns:
                Predicted values of the trained Naive Bayes model.
    """

    def naive_bayes(self) -> np.ndarray:
        self.nbModel = MultinomialNB()
        self.nbModel.fit(self.x_train, self.y_train)

        return self.nbModel.predict(self.x_test)
    
    def predict_naive_bayes(self, inputResumesDf) -> np.ndarray:
        return self.nbModel.predict(inputResumesDf)

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

    def read_file_content(self,file):
        
        file.save(os.path.join('data/resumes', file.filename))
        originalFilePath = 'data/resumes/'+file.filename

        match file.content_type:
            case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                text = textract.process(originalFilePath)
                print(text)
                content = text.decode("utf-8")
                return file.filename, content
            case "image/png":
                newFileName = file.filename.split('.')[0] + ".pdf"
                ImageToPdfConverter.ImageToPdf(originalFilePath).SaveAs('data/resumes/'+newFileName)
                pdf = PdfDocument.FromFile('data/resumes/'+newFileName)
                all_text = pdf.ExtractAllText()
                print(all_text)
                return file.filename, all_text
            case "application/pdf":
                pdf = PdfDocument.FromFile(originalFilePath)
                all_text = pdf.ExtractAllText()
                print(all_text)
                return file.filename, all_text
            case _:
                file.seek(0, 0)
                content = file.readlines()
                content = b" ".join(content)
                content = content.decode("utf-8")
                return file.filename, content
    
    @staticmethod
    def classify_resume(self, inputResumes, predictionModel):

        resumeContent =  [self.read_file_content(x) for x in inputResumes]
        inputResumesDf = pd.DataFrame(resumeContent, columns =['inputFileName','inputFileContent'])
        
        inputResumesDfOutput = self.wordProcessor.setup_resume_data(inputResumesDf)

        if(predictionModel == "Logistic Regression (Recommended)"):
            return self.lb_encd.inverse_transform(self.predict_logistic_regression(inputResumesDfOutput))
        if(predictionModel == "Support Vector Machine"):
            return self.lb_encd.inverse_transform(self.predict_svm(inputResumesDfOutput))
        if(predictionModel == "Naive Bayes"):
            return self.lb_encd.inverse_transform(self.predict_naive_bayes(inputResumesDfOutput))
        else:
            return self.lb_encd.inverse_transform(self.predict_logistic_regression(inputResumesDfOutput))

    @staticmethod
    def load_train_models(self):
        # Example usage
        # Generate synthetic data for demonstration
        # X_sparse = csr_matrix(self.x)
        
        # Perform train-test split
        # self.x_train, self.x_test, self.y_train, self.y_test = self.wordProcessor.custom_train_test_split(
        #     x = self.x,
        #     y = self.y
        # )

        self.x_train, self.x_test, self.y_train, self.y_test, self.lb_encd = self.wordProcessor.setup_data(
            self.rawData
        )

        # X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
        # X_sparse = csr_matrix(X)
        
        # Instantiate the Models class
        # global models_instance
        # models_instance = Model(X_sparse, y)

        # Train and evaluate Logistic Regression
        logistic_regression_predictions = self.logistic_regression()
        accuracy_lr, report_lr = self.get_classification_report(self.y_test, logistic_regression_predictions)

        print("Logistic Regression Accuracy:", accuracy_lr)
        print("Logistic Regression Classification Report:\n", report_lr)

        # # Train and evaluate Support Vector Machine
        svm_predictions = self.svm()
        accuracy_svm, report_svm = self.get_classification_report(self.y_test, svm_predictions)

        print("\nSVM Accuracy:", accuracy_svm)
        print("SVM Classification Report:\n", report_svm)

        # # Similarly, train and evaluate other models...

        # # Train and evaluate Naive Bayes
        nb_predictions = self.naive_bayes()
        accuracy_nb, report_nb = self.get_classification_report(self.y_test, nb_predictions)

        print("\nNaive Bayes Accuracy:", accuracy_nb)
        print("Naive Bayes Classification Report:\n", report_nb)

        # return models_instance
    