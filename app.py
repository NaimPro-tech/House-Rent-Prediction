from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

#load artifacts
data = joblib.load("artifacts.pkl")
model = data['model']
encoder = data['target_encoder']
features = data['features']

@app.route("/")
def home():
    return "Welcome to House Rent Prediction"

#prediction API

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.json

        df = pd.DataFrame([input_data])

        #encoding
        df['area_locality'] = encoder.transform(df['area_locality'])

        #one hot (same as training)
        df = pd.get_dummies(df)

        #match columns
        df = df.reindex(columns=features, fill_value=0)

        prediction = model.predict(df)

        #inverse log transform to original scale
        prediction = np.expm1(prediction)
        
        return jsonify({
            "Predict_rent":float(prediction[0])
        })
    except Exception as e:
        return jsonify({
            "Error": str(e)
        })




if __name__ == "__main__":
    app.run(debug=True)