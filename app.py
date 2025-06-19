# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

# Load the trained model 
model_path = 'best_model.pkl'
with open(model_path,'rb') as file:
    model = pickle.load(file)

encoder_path = 'label_encoders.pkl'
with open(encoder_path,'rb') as file:
    encoder = pickle.load(file)    

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final_features = list(request.form.keys())
    input_dict = dict(zip(final_features,int_features))
    data_frame = pd.DataFrame(input_dict, index=[1])

    # separating cat_features
    dataframe_cat = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
    
    for col in dataframe_cat:
     data_frame[col] = encoder[col].transform(data_frame[col])

    # make prediction
    preds = model.predict(data_frame)
    if preds==1:
        counter = 'YES patience have Heart Disease'
    else:
        counter = 'NO, patience does not have heart Disease'   

    return render_template('index.html',prediction_text='prediction : {}'.format(counter))
    
if __name__=="__main__":
    app.run(debug=True) 