from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# Importar los modelos
random_forest = pickle.load(open('model.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))


def prediction( gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialisation, mba_p):
    # Crear un DataFrame con los datos de entrada
    input_data = pd.DataFrame({
        'gender': [gender],
        'ssc_p': [ssc_p],
        'hsc_p': [hsc_p],
        'degree_p': [degree_p],
        'workex': [workex],
        'etest_p': [etest_p],
        'specialisation': [specialisation],
        'mba_p': [mba_p]
    })
        
    input_data['gender'] = input_data['gender'].map({'M': 0, 'F': 1})
    
    input_data['workex'] = input_data['workex'].map({'No': 0, 'Yes': 1})
    
    input_data['specialisation'] = input_data['specialisation'].map({'Mkt&HR': 0, 'Mkt&Fin': 1})
    

    input_data_scaled = ms.transform(input_data)
    
    prediction = random_forest.predict(input_data_scaled)
    
    
    return prediction[0]

# Crear Flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():

  
    gender = request.form['gender']
    ssc_p = request.form['ssc_p']
    hsc_p = request.form['hsc_p']
    degree_p = request.form['degree_p']
    workex = request.form['workex']
    etest_p = request.form['etest_p']
    specialisation = request.form['specialisation']
    mba_p = request.form['mba_p']

    # Predecir el cluster
    resultpre = prediction( gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialisation, mba_p)
    if resultpre == 1:
        resultpre = 'contratado'
    else:
        resultpre ='No Contratado'

    return render_template('index.html', result="El postulante debe ser  {} ".format(resultpre))


# Python main
if __name__ == "__main__":
    app.run(debug=True, port=8080, host="0.0.0.0")

