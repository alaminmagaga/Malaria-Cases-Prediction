import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Create flask app
app = Flask(__name__)

# Load the pickle model
model = pickle.load(open('malaria.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/data')
def data():
    # Load the entire dataset from a CSV file
    dataset_path = 'DatasetAfricaMalaria.csv'
    df = pd.read_csv(dataset_path)
    df_head = df.head()
    return render_template('dataset.html', df_head=df_head)

@app.route("/predict", methods=["POST"])
def predict():
       
        # Extract features from the request form
        year = float(request.form.get('year'))
        iom = float(request.form.get('iom'))
        ruralpop = float(request.form.get('ruralpop'))
        ruralpopgrowth = float(request.form.get('ruralpopgrowth'))
        urbanpop = float(request.form.get('urbanpop'))
        urbanpopgrowth = float(request.form.get('urbanpopgrowth'))
        basicdrinkingwaterservices = float(request.form.get('basicdrinkingwaterservices'))
        basicdrinkingwater_rural = float(request.form.get('basicdrinkingwater_rural'))
        basicdrinkingwater_urban = float(request.form.get('basicdrinkingwater_urban'))
        bascisanitationservices = float(request.form.get('bascisanitationservices'))
        bascisanitationservices_rural = float(request.form.get('bascisanitationservices_rural'))
        bascisanitationservices_urban = float(request.form.get('bascisanitationservices_urban'))
        latitude = float(request.form.get('latitude'))
        longitude = float(request.form.get('longitude'))

        # Create feature array for prediction
        features = np.array([[
            year, iom, ruralpop, ruralpopgrowth, urbanpop,
            urbanpopgrowth, basicdrinkingwaterservices, basicdrinkingwater_rural,
            basicdrinkingwater_urban, bascisanitationservices, bascisanitationservices_rural,
            bascisanitationservices_urban, latitude, longitude
        ]])

        # Make prediction using the loaded model
        prediction = model.predict(features)
        # Render the result template with only the prediction value
        return render_template('result.html', prediction=prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
