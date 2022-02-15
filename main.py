import pandas as pd
import numpy as np
import pickle
from flask import Flask,render_template,request
app=Flask(__name__,template_folder='template')
model=pickle.load(open(r'emppre.pkl', 'rb'))
data=pd.read_csv('train2.csv')

import pandas as pd

def train2(data):
    assert isinstance(data, pd.DataFrame), "df needs to be a pd.DataFrame"
    data.dropna(inplace=True)
    indices_to_keep = data.isin([np.nan, np.inf, -np.inf]).any(1)
    return data[indices_to_keep].astype(np.float64)


@app.route('/')
def index():

    department = sorted(data['department'].unique())
    education= sorted(data['education'].unique())
    gender= sorted(data['gender'].unique())
    no_of_trainings=sorted(data['no_of_trainings'].unique())
    previous_year_rating=sorted(data['previous_year_rating'].unique())

    department.insert(0,'Select Your Department')
    education.insert(0,'Select Your Education')
    gender.insert(0,'Select Your Gender')
    no_of_trainings.insert(0,'No Of Trainings')
    previous_year_rating.insert(0,'What is your previous year rating')

    return render_template('index.html',department=department,education=education,gender=gender,no_of_trainings=no_of_trainings,previous_year_rating=previous_year_rating )

@app.route('/predicts',methods=['POST'])
def predict():
    department=request.form.get('department')
    education =request.form.get('education')
    gender =request.form.get('gender')
    no_of_trainings=request.form.get('no_of_trainings')
    age=request.form.get('age')
    previous_year_rating=request.form.get('previous_year_rating')
    length_of_service=request.form.get('length_of_service')
    KPIs_greater_than_80=request.form.get('KPIs_greater_than_80')
    awards_won = request.form.get('awards_won')
    avg_training_score=request.form.get('avg_training_score')
    input=pd.DataFrame([[department,education,gender,no_of_trainings,age,previous_year_rating,length_of_service,KPIs_greater_than_80,awards_won,avg_training_score]],columns=['department','education','gender','no_of_trainings','age','previous_year_rating','length_of_service','KPIs_greater_than_80','awards_won','avg_training_score'])
    prediction = model.predict(input)
    return str(prediction)

if __name__=='__main__':
    app.run(debug=True,port=5001)