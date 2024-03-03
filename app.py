#import the relevant libraries
import os
from flask import Flask, flash, request, render_template
from cs50 import SQL
import csv
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')
#configure application
app = Flask(__name__)

db = SQL("sqlite:///data.db")

data = pd.read_csv("Heart_Disease_Prediction.csv")
#Explore the dataset
data.head()
data.describe()

y= data['Heart Disease'] #dependent variable is Decision
x= data.drop(['Heart Disease'], axis=1)

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2)

#Implementing Logistic Regression using sklearn
modelLogistic = LogisticRegression()
modelLogistic.fit(x_train,y_train)

#Make prediction for the test data
y_pred= modelLogistic.predict(x_test)

#Creating confusion matrix
ConfusionMatrix = confusion_matrix(y_test, y_pred)

#Accuracy from confusion matrix
TP= ConfusionMatrix[1,1] #True positive
TN= ConfusionMatrix[0,0] #True negative
Total=len(y_test)

def checker(sample):
    prediction = modelLogistic.predict(sample)
    return prediction[-1]



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        age = request.form.get("age")
        sex = request.form.get("sex")
        chest_pain_type = request.form.get("chest_pain_type")
        bp = request.form.get("BP")
        cholesterol = request.form.get("cholesterol")
        FBS_over_120 = request.form.get("FBS_over_120")
        EKGresult = request.form.get("EKG_Result")
        max_HR = request.form.get("max_HR")
        exercise_angina = request.form.get("exercise_angina")
        ST_depression = request.form.get("ST_depression")
        slope_of_ST = request.form.get("slope_of_ST")
        number_of_vessels_fluoro = request.form.get("number_of_vessels_fluoro")
        thallium = request.form.get("thallium")

        db.execute("INSERT INTO vitals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", age, sex, chest_pain_type, bp, cholesterol, FBS_over_120, EKGresult, max_HR, exercise_angina, ST_depression, slope_of_ST, number_of_vessels_fluoro, thallium)

        vitals = db.execute("SELECT * FROM vitals")
        vitals = vitals[0]
        vital_list = []
        for vital in vitals:
            vital_list.append(vitals[vital])

        file = open("vitals.csv", "a")
        writer = csv.writer(file)
        writer.writerow(vital_list)
        file.close()

        df = pd.read_csv("vitals.csv")

        result = checker(df)
        db.execute("DELETE FROM vitals")
        return render_template("report.html", result=result)
    else:
        return render_template("index.html")
    

