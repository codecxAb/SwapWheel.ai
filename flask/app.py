from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open("../carModel.pkl", 'rb'))

car = pd.read_csv("../cleaned_car.csv")

@app.route('/')
def hello_world():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    car_years = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    return render_template('index.html', companies=companies, car_models=car_models, car_years=car_years, fuel_type=fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('model')
        car_year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel')
        kms = int(request.form.get('kms'))

        # Validate input data
        if company not in car['company'].unique():
            return "Invalid company", 400
        if car_model not in car['name'].unique():
            return "Invalid car model", 400
        if car_year not in car['year'].unique():
            return "Invalid car year", 400
        if fuel_type not in car['fuel_type'].unique():
            return "Invalid fuel type", 400

        # Prepare the input data for the model
        input_data = pd.DataFrame([[car_model, company, car_year, fuel_type, kms]], 
                                  columns=['name', 'company', 'year', 'fuel_type', 'kms_driven'])
        prediction = model.predict(input_data)
        predicted_price = prediction[0]
        
        return str(round(predicted_price,2))
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred during prediction", 500

if __name__ == "__main__":
    app.run(debug=True)
