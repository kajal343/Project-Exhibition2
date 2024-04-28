import joblib
import pandas as pd
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify

df = pd.read_csv(r"C:\Users\hp\Documents\Dataset_Crop Recomendation.csv")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Predict')
def prediction():
    return render_template('index.html')

@app.route('/component')
def component():
    return render_template('component.html')

@app.route('/form', methods=["POST"])
def form_handler():
    city = request.form['city']  # Extract city from form data
    temperature, humidity = get_weather_data(city)  # Call get_weather_data with city name
    
    # Check if weather data is fetched successfully
    if temperature is not None and humidity is not None:
        Nitrogen = float(request.form['Nitrogen'])
        Phosphorus = float(request.form['Phosphorus'])
        Potassium = float(request.form['Potassium'])
        Ph = float(request.form['ph'])
        Rainfall = float(request.form['Rainfall'])
        
        prediction_result = brain(Nitrogen, Phosphorus, Potassium, temperature, humidity, Ph, Rainfall)
        return render_template('prediction.html', prediction=prediction_result['prediction'], ml_link=prediction_result['ml_link'])
    else:
        return "Failed to fetch weather data for the specified city"

def get_weather_data(city):
    api_key = '632c9c8733727bc87ef366a47fda17d4'
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'main' in data:
            temp = data['main'].get('temp')
            humidity = data['main'].get('humidity')
            return temp, humidity
        else:
            print("Key 'main' not found in weather data")
            return None, None
    else:
        print("Failed to fetch weather data")
        return None, None

def brain(Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Ph, Rainfall):
    values = [Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Ph, Rainfall]
    if Ph > 0 and Ph <= 14 and Temperature < 100 and Humidity > 0:
        model = joblib.load(open('crop app', 'rb'))
        arr = [values]
        acc = model.predict(arr)
        input_data = values
        input_data = np.array(input_data).reshape(1, -1)
        predicted_crop_label = model.predict(input_data)[0]
        predicted_crop_row = df.loc[df['label'] == predicted_crop_label]
        youtube_link = predicted_crop_row['youtube link'].values[0]
        return {'prediction': str(acc), 'ml_link': youtube_link}
    else:
        return "Sorry... Error in entered values in the form. Please check the values and fill it again"

if __name__ == '__main__':
    app.run(debug=True)

