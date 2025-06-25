from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)
model = joblib.load('model/model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    area = int(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathroom = int(request.form['bathroom'])  # from input name="bathrooms"
    stories = int(request.form['stories'])
    mainroad = request.form['mainroad']
    guestroom = request.form['guestroom']
    basement = request.form['basement']
    hotwaterheating = request.form['hotwaterheating']  # corrected variable name
    airconditioning = request.form['airconditioning']
    parking = int(request.form['parking'])
    prefarea = request.form['prefarea']
    furnishingstatus = request.form['furnishingstatus']

    # Create a single-row DataFrame matching the model's columns
    data = pd.DataFrame([{
        'area': area,
        'bedrooms': bedrooms,
        'bathrooms': bathroom,
        'stories': stories,
        'mainroad': mainroad,
        'guestroom': guestroom,
        'basement': basement,
        'hotwaterheating': hotwaterheating,
        'airconditioning': airconditioning,
        'parking': parking,
        'prefarea': prefarea,
        'furnishingstatus': furnishingstatus
    }])

    # Make prediction
    prediction = model.predict(data)[0]

    return render_template('index.html', prediction=round(prediction, 2))


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

    # app.run(debug=True)
