from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the input values from the form
            data = {
                'Gender': request.form['Gender'],
                'Married': request.form['Married'],
                'Dependents': request.form['Dependents'],
                'Education': request.form['Education'],
                'Self_Employed': request.form['Self_Employed'],
                'ApplicantIncome': float(request.form['ApplicantIncome']),
                'CoapplicantIncome': float(request.form['CoapplicantIncome']),
                'LoanAmount': float(request.form['LoanAmount']),
                'Loan_Amount_Term': float(request.form['Loan_Amount_Term']),
                'Credit_History': float(request.form['Credit_History']),
                'Property_Area': request.form['Property_Area']
            }

            # Convert the input values to a numpy array
            input_data = np.array([[data['Gender'], data['Married'], data['Dependents'], data['Education'], 
                                    data['Self_Employed'], data['ApplicantIncome'], data['CoapplicantIncome'], 
                                    data['LoanAmount'], data['Loan_Amount_Term'], data['Credit_History'], 
                                    data['Property_Area']]])

            # Make prediction
            prediction = model.predict(input_data)

            # Calculate estimated loan amount and interest rate if not approved
            estimated_loan = None
            interest_rate = None
            if prediction[0] == 0:  # Loan not approved
                estimated_loan = ((data['ApplicantIncome']+data['CoapplicantIncome']) * 5);  # Example: Loan up to 5 times income
                interest_rate = 10 + (100000 - estimated_loan) / 10000  # Example: Higher rate for smaller loans

            return render_template('after.html', 
                                   prediction=prediction[0],
                                   estimated_loan=estimated_loan, 
                                   interest_rate=interest_rate)
        except Exception as e:
            return str(e)

if __name__ == '__main__':
    app.run(debug=True)