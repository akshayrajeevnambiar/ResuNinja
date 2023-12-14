import os
from flask import Flask, render_template, request
from models.model import Model
import pandas as pd

app = Flask(__name__)
models = Model()

app.config['UPLOAD_EXTENSIONS'] = ['.txt', '.docx', '.pdf']

@app.route('/predict-resume', methods=['GET', 'POST'])
def index():
    uploaded_files = []
    if request.method == 'POST':
        
        prediction_model = request.form.get('predictionModel')
        uploaded_files = request.files.getlist("fileInput[]")
        
        filesDf = pd.DataFrame(map(lambda x: x.filename, uploaded_files),columns=['fileName'])
        
        availableCategories = models.rawData['Category'].unique()

        if(len(filesDf) > 0):

            if uploaded_files:
                predictions = models.classify_resume(models, uploaded_files, prediction_model)
                
                predictionDf = pd.DataFrame(predictions,columns=['prediction'])
                
                uploaded_files = []
                return render_template('index.html', countOfAvailableCategories = len(availableCategories),available_Categories=availableCategories, prediction_model=prediction_model, predictions=True,file_data=filesDf,prediction_data = predictionDf, countOfUploadedResumes = len(predictions))
        else:
            uploaded_files = []
            return render_template('index.html', errors="Please upload valid files for resume classification. (eg. .txt)")

    return render_template('index.html',input_data = uploaded_files)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)