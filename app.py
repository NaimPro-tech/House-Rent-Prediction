from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from utils import create_features, preprocess

app = Flask(__name__)

#load artifacts
data = joblib.load("artifacts.pkl")
model = data['model']
encoder = data['target_encoder']
features = data['features']

@app.route("/")
def home():
    return render_template("index.html")

#prediction API

@app.route("/predict_ui", methods=["POST"])
def predict_ui():
    try:
        input_data = request.form.to_dict()

        #convert numeric field
        int_cols = ["bhk", "bathroom", "living_floor", "total_floor"]
        float_cols = ["size"]

        for col in int_cols:
            if col in input_data:
                input_data[col] = int(input_data[col])

        for col in float_cols:
            if col in input_data:
                input_data[col] = float(input_data[col])

        df = pd.DataFrame([input_data])
        df = create_features(df=df)

        # #encoding
        # df['area_locality'] = encoder.transform(df['area_locality'])

        # #one hot (same as training)
        # df = pd.get_dummies(df)

        # #match columns
        # df = df.reindex(columns=features, fill_value=0)

        df = preprocess(df=df, encoder=encoder, features=features)

        prediction = model.predict(df)

        #inverse log transform to original scale
        prediction = np.expm1(prediction)
        
        return render_template(
            "index.html",
            prediction_text = f"Predicted Rent: {round(prediction[0],2)}"
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text = f"Error: {str(e)}"
        )




if __name__ == "__main__":
    app.run(debug=True)