from flask import Flask, render_template , request 
import pickle as pk 
import numpy as np 


# Load the pre-trained model
model = pk.load(open('churn_logistic_model.pkl', 'rb'  ))
scaler = pk.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods = ['POST', 'GET'])
def predict():
        if request.method == 'POST':
            gender = float(request.form['gender'])
            SeniorCitizen = float(request.form['SeniorCitizen'])
            Partner = float(request.form['Partner'])
            Dependents = float(request.form['Dependents'])
            tenure = float(request.form['tenure'])
            phone = float(request.form['PhoneService'])
            multiple_lines = float(request.form['MultipleLines'])
            contract = float(request.form['Contract'])
            total_charges = request.form['TotalCharges']


          # Convert TotalCharges to float; fill if empty or invalid
            try:
                total_charges = float(total_charges)
            except ValueError:
            # If conversion fails (empty or non-numeric), use 0 or some default
                    total_charges = 0.0

        # create a feature array 
            features = np.array([gender,SeniorCitizen, Partner, Dependents,
                              tenure, phone, multiple_lines,
                              contract, total_charges])

        
        #scale the features 
            feature_scaled = scaler.transform([features])

        # make prediction 
            prediction = model.predict(feature_scaled)[0]
            output = 'Churn' if prediction == 1 else 'Not Churn'

            return render_template('index.html', prediction = output)
        else:
        # get request : show the form 
            return render_template('index.html', prediction = 'Please fill the form')
    

if __name__ == '__main__':
    app.run(debug = True)