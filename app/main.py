import os
from flask import Flask, render_template, request
from models.model import Model

app = Flask(__name__)
models = Model

@app.route('/predict-resume', methods=['GET', 'POST'])
def index():
    input_data = []
    if request.method == 'POST':
        input_data = [
            request.form.get('input_resumes')]

        if input_data:
            prediction = models.classify_resume(models, input_data)
            return render_template('index.html', prediction=prediction,input_data = input_data)

    return render_template('index.html',input_data = input_data)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)