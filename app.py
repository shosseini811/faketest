# this file takes input from the html file, uses the input to run the serialized model (Pickled model), and then 
# outputs the target information

from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
import joblib
import nltk
from joblib import load, dump
from sklearn import preprocessing
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')

# load dataset into dataframe object 
#df_true = pd.read_csv("https://final-project-data-rjj.s3.us-east-2.amazonaws.com/True.csv")
#df_fake = pd.read_csv("https://final-project-data-rjj.s3.us-east-2.amazonaws.com/Fake.csv")
    
#df_true['target'] = "true"  
#df_fake['target'] = "fake"
#df = pd.concat([df_true, df_fake]).reset_index(drop = True)

# initialize new Flask instance with argument __name__ 
app = Flask(__name__)
model = pickle.load(open('trained_regression_model.pkl', 'rb'))

# ROUTES
@app.route('/')
def home():
    return render_template('home.html')

# within the predict function, we do the following: access the True and Fake datasets, pre-process
# the text, make predictions, store the model, access message entered by the user, use our model 
# to make prediction for its label
# the POST method transports form data to the server in the message body

@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method == 'POST':
        par_text = request.form['news_paragraph']
        stop_words = stopwords.words('english')
        stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
        input_test2 = [par_text]
        input_output = [word for word in input_test2 if not word in stop_words]
        news_prediction = model.predict(input_output)   
 
        return render_template('home.html', prediction = news_prediction)

import logging
import sys
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

# the if __name == '__main__' statement ensure that the run function will only run the application 
# on the server when the script is directly executed by Python interpreter
if __name__ == '__main__':
    app.run(debug=True)

