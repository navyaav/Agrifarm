from flask import Flask,request,jsonify
import pickle
import numpy as np
model=pickle.load(open('RandomForest.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return "Hello world"

@app.route('/predict',methods=['POST'])
def predict():
    n=request.form.get('n')
    p=request.form.get('p')
    k=request.form.get('k')
    temp=request.form.get('temp')
    humi=request.form.get('humi')
    ph=request.form.get('ph')
    rain=request.form.get('rain')

    input_query=np.array([[n,p,k,temp,humi,ph,rain]])
    result=model.predict(input_query)[0]
    return jsonify({'crop':str(result)})


if __name__=='__main__':
    app.run(debug=True)
