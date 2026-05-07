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
        
        #add input validation
        required_fields = [
            "area_locality",
            "bhk",
            "size",
            "bathroom",
            "living_floor",
            "total_floor"
        ]
        for field in required_fields:
            if field not in input_data or input_data[field]=="":
                return render_template(
                    "index.html",
                    prediction_text = f"{field} is required!"
                )
        
        #numeric validation
        if input_data["bhk"]<=0:
            return render_template(
                "index.html",
                prediction_text = "BHK must be greater than Zero"
            )

        if input_data["size"]<=0:
            return render_template(
                "index.html",
                prediction_text = "How can you live without space in your house!!!"
            )
        
        if input_data["bathroom"]<=0:
            return render_template(
                "index.html",
                prediction_text = "Your home should have Bathroom."
            )

        if input_data["living_floor"]>input_data["total_floor"]:
            return render_template(
                "index.html",
                prediction_text = "Your are too above from your building."
            )
        
        df = pd.DataFrame([input_data])
        df = create_features(df=df)


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
            prediction_text = f"Something went wrong!: {str(e)}"
        )




if __name__ == "__main__":
    app.run(debug=True)