import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from ensemble import WeightedVotingRegressor # Need this class definition importable

app = Flask(__name__, static_folder="../public", static_url_path="")
CORS(app)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Global dict to store loaded models and preprocessors
loaded_assets = {}

def load_ml_assets():
    print("Loading ML models and preprocessors into memory...")
    try:
        # Load preprocessors
        with open(os.path.join(MODELS_DIR, "dataset_1_preprocessors.pkl"), "rb") as f:
            loaded_assets["preproc_1"] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "dataset_2_preprocessors.pkl"), "rb") as f:
            loaded_assets["preproc_2"] = pickle.load(f)
            
        # Load models
        with open(os.path.join(MODELS_DIR, "models_d1_all.pkl"), "rb") as f:
            loaded_assets["models_d1_all"] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "models_d1_fs.pkl"), "rb") as f:
            loaded_assets["models_d1_fs"] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "models_d2_all.pkl"), "rb") as f:
            loaded_assets["models_d2_all"] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "models_d2_fs.pkl"), "rb") as f:
            loaded_assets["models_d2_fs"] = pickle.load(f)
            
        # Dynamically fit LIME explainers on startup to avoid pickling lambda issues
        from preprocess import preprocess_dataset_1, preprocess_dataset_2
        import lime.lime_tabular
        
        print("  Dynamically fitting LIME explainers on startup...")
        X_train_1, _, _, _, _, _, _, _ = preprocess_dataset_1()
        X_train_2, _, _, _, _, _ = preprocess_dataset_2()
        
        loaded_assets["lime_d1"] = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_1,
            feature_names=loaded_assets["preproc_1"]['feature_cols'],
            class_names=['Performance Index'],
            mode='regression',
            random_state=42
        )
        
        loaded_assets["lime_d2"] = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_2,
            feature_names=loaded_assets["preproc_2"]['feature_cols'],
            class_names=['Exam Score'],
            mode='regression',
            random_state=42
        )
        print("  LIME explainers fitted successfully.")
            
        # Load metrics
        with open(os.path.join(MODELS_DIR, "metrics.json"), "r") as f:
            loaded_assets["metrics"] = json.load(f)
            
        # Load SHAP global importance
        with open(os.path.join(MODELS_DIR, "shap_importance.json"), "r") as f:
            loaded_assets["shap_importance"] = json.load(f)
            
        print("All assets loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading ML assets: {e}")
        return False

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    if "metrics" not in loaded_assets:
        return jsonify({"error": "Assets not loaded"}), 500
    return jsonify(loaded_assets["metrics"])

@app.route("/api/shap", methods=["GET"])
def get_shap():
    if "shap_importance" not in loaded_assets:
        return jsonify({"error": "Assets not loaded"}), 500
    return jsonify(loaded_assets["shap_importance"])

@app.route("/api/predict", methods=["POST"])
def predict():
    if not loaded_assets:
        return jsonify({"error": "Models are not loaded yet"}), 500
        
    data = request.get_json()
    dataset_id = data.get("dataset_id", "dataset_1")
    feature_selection = data.get("feature_selection", False)
    raw_inputs = data.get("features", {})
    
    try:
        if dataset_id == "dataset_1":
            preproc = loaded_assets["preproc_1"]
            feature_cols = preproc["feature_cols"]
            
            # Map input to dataframe
            input_df = pd.DataFrame([raw_inputs])
            
            # 1. Encode Extracurricular Activities
            le = preproc["label_encoder"]
            if 'Extracurricular Activities' in input_df.columns:
                val = input_df['Extracurricular Activities'].iloc[0]
                if isinstance(val, str):
                    input_df['Extracurricular Activities'] = le.transform([val])
                else:
                    input_df['Extracurricular Activities'] = int(val)
            
            # Order features correctly
            input_df = input_df[feature_cols]
            
            # 2. Scale features
            scaler_x = preproc["scaler_x"]
            X_scaled = scaler_x.transform(input_df)
            
            # Select model & preprocess subset if feature selected
            if feature_selection:
                model = loaded_assets["models_d1_fs"]["Ensemble VR"]
                X_input = X_scaled[:, preproc["fs_indices"]]
            else:
                model = loaded_assets["models_d1_all"]["Ensemble VR"]
                X_input = X_scaled
                
            # Make prediction (scaled)
            pred_scaled = model.predict(X_input)[0]
            
            # Inverse scale target to original scale
            scaler_y = preproc["scaler_y"]
            pred_original = scaler_y.inverse_transform([[pred_scaled]])[0][0]
            
            # LIME local explanation
            # For explanation we use the full model, so we explain on X_scaled
            explainer = loaded_assets["lime_d1"]
            full_model_for_lime = loaded_assets["models_d1_all"]["Ensemble VR"]
            
            exp = explainer.explain_instance(
                data_row=X_scaled[0],
                predict_fn=full_model_for_lime.predict,
                num_features=len(feature_cols)
            )
            
            lime_list = exp.as_list()
            # Format LIME output to represent feature and its contribution
            lime_contributions = []
            for feat_rule, weight in lime_list:
                # Find which feature name is in the rule
                feat_name = None
                for col in feature_cols:
                    if col in feat_rule:
                        feat_name = col
                        break
                lime_contributions.append({
                    "rule": feat_rule,
                    "feature": feat_name or feat_rule,
                    "contribution": float(weight)
                })
                
            return jsonify({
                "prediction": float(pred_original),
                "prediction_scaled": float(pred_scaled),
                "lime": lime_contributions
            })
            
        elif dataset_id == "dataset_2":
            preproc = loaded_assets["preproc_2"]
            feature_cols = preproc["feature_cols"]
            
            # Map input to dataframe
            input_df = pd.DataFrame([raw_inputs])
            
            # Fill missing categorical values with mode, numerical with mean
            for col in preproc["modes"]:
                if col not in input_df.columns or pd.isna(input_df[col].iloc[0]):
                    input_df[col] = preproc["modes"][col]
            for col in preproc["means"]:
                if col not in input_df.columns or pd.isna(input_df[col].iloc[0]):
                    input_df[col] = preproc["means"][col]
                    
            # Encode categorical
            for col, mapping in preproc["ordinal_mappings"].items():
                val = input_df[col].iloc[0]
                try:
                    if val in mapping:
                        input_df[col] = mapping[val]
                    else:
                        input_df[col] = int(val)
                except Exception:
                    input_df[col] = 0 # Fallback
                    
            # Order features correctly
            input_df = input_df[feature_cols]
            
            # Scale features
            scaler_x = preproc["scaler_x"]
            X_scaled = scaler_x.transform(input_df)
            
            # Select model
            if feature_selection:
                model = loaded_assets["models_d2_fs"]["Ensemble VR"]
                X_input = X_scaled[:, preproc["fs_indices"]]
            else:
                model = loaded_assets["models_d2_all"]["Ensemble VR"]
                X_input = X_scaled
                
            # Make prediction (original scale)
            pred = model.predict(X_input)[0]
            
            # LIME local explanation
            explainer = loaded_assets["lime_d2"]
            full_model_for_lime = loaded_assets["models_d2_all"]["Ensemble VR"]
            
            exp = explainer.explain_instance(
                data_row=X_scaled[0],
                predict_fn=full_model_for_lime.predict,
                num_features=len(feature_cols)
            )
            
            lime_list = exp.as_list()
            lime_contributions = []
            for feat_rule, weight in lime_list:
                feat_name = None
                for col in feature_cols:
                    if col in feat_rule:
                        feat_name = col
                        break
                lime_contributions.append({
                    "rule": feat_rule,
                    "feature": feat_name or feat_rule,
                    "contribution": float(weight)
                })
                
            return jsonify({
                "prediction": float(pred),
                "lime": lime_contributions
            })
            
        else:
            return jsonify({"error": f"Unknown dataset ID: {dataset_id}"}), 400
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Check if files exist before running, otherwise load later
    load_ml_assets()
    app.run(host="0.0.0.0", port=5000, debug=True)
