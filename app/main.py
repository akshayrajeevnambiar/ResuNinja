from flask import Flask, render_template, request
from app.models import Models
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Load your machine learning model from app.models
    # Replace this with your actual model loading code
    x, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
    x_sparse = csr_matrix(x)
    models_instance = Models(x_sparse, y)

    # Get input data from the form
    # Replace 'input_name' with actual input field name from your HTML form
    input_data = request.form['input_name']

    # Perform prediction using your model
    # Replace 'prediction' with your actual prediction code
    prediction = models_instance.logistic_regression()

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
