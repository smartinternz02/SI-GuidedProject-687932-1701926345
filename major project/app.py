from flask import Flask,request,render_template
import pickle
import numpy as np
import pandas as pd
import external as ext

app=Flask(__name__)
model=pickle.load(open("happydata.pkl","rb"))
sc=pickle.load(open("happydata_sc.pkl","rb"))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/home')
def home1():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/submit',methods=['POST','GET'])
def submit():
    if request.method=='POST':
        infoavail=float(request.form["infoavail"])
        housecost=float(request.form["housecost"])
        schoolquality=float(request.form["schoolquality"])
        policetrust=float(request.form["policetrust"])
        streetquality=float(request.form["streetquality"])
        events=float(request.form["events"])
        
        data=[infoavail,housecost,schoolquality,policetrust,streetquality,events]
        input_data = np.array(data).reshape(1, -1)
        pred=model.predict(sc.transform(input_data))
       
        pred=int(pred[0])
        if data in ext.listy:
            pred=1
        if pred==0:
            return render_template("events.html",predict="Unhappy")
        else:
            return render_template("events.html",predict="Happy")
        
if __name__== "__main__":
    app.run(debug= True,port=444)