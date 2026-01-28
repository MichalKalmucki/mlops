
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import torch
from model import CarbonModel

preprocessor = joblib.load("preprocessor.joblib")

input_dim = 21
model = CarbonModel(input_dim=input_dim, hidden_dim=64)
model.load_state_dict(torch.load("carbon_model.pth"))
model.eval()


categorical_cols = ["day_type", "transport_mode", "food_type", "eco_actions"]
numerical_cols = ["distance_km", "electricity_kwh", "renewable_usage_pct", "screen_time_hours", "waste_generated_kg"]
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])

    for col in categorical_cols + numerical_cols:
        if col not in df.columns:
            df[col] = 0  # Fill missing

    X = preprocessor.transform(df)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        pred = model(X_tensor)

    return jsonify({"carbon_footprint_kg": float(pred[0])})

if __name__ == "__main__":
    app.run(port=5000)
