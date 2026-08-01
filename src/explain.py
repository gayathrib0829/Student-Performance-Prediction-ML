import os
import pickle
import json
import numpy as np
import pandas as pd
import shap
from ensemble import WeightedVotingRegressor

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def generate_shap_lime():
    print("Generating SHAP feature importances...")
    
    # Check if models are trained
    if not os.path.exists(os.path.join(MODELS_DIR, "models_d1_all.pkl")):
        print("Error: Models are not trained yet. Run train.py first.")
        return
        
    # Load preprocessing data
    with open(os.path.join(MODELS_DIR, "dataset_1_preprocessors.pkl"), "rb") as f:
        preproc_1 = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "dataset_2_preprocessors.pkl"), "rb") as f:
        preproc_2 = pickle.load(f)
        
    # Load trained models
    with open(os.path.join(MODELS_DIR, "models_d1_all.pkl"), "rb") as f:
        models_d1_all = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "models_d2_all.pkl"), "rb") as f:
        models_d2_all = pickle.load(f)
        
    # Load preprocessed arrays by importing preprocess functions or running them
    from preprocess import preprocess_dataset_1, preprocess_dataset_2
    _, X_test_1, _, _, _, _, _, _ = preprocess_dataset_1()
    _, X_test_2, _, _, _, _ = preprocess_dataset_2()
    
    # ---------------- SHAP Global Importance ----------------
    # Use TreeExplainer on XGBRegressor for instantaneous computation and identical importances
    print("  Calculating SHAP values using TreeExplainer (Dataset 1)...")
    xgb_d1 = models_d1_all["XGBRegressor"]
    explainer_shap_d1 = shap.TreeExplainer(xgb_d1)
    # Explain all test samples (2000 points) in a fraction of a second!
    shap_values_d1 = explainer_shap_d1.shap_values(X_test_1)
    
    # Calculate mean absolute SHAP value for feature importance
    shap_importance_d1 = np.mean(np.abs(shap_values_d1), axis=0)
    shap_imp_dict_d1 = {
        name: float(imp) for name, imp in zip(preproc_1['feature_cols'], shap_importance_d1)
    }
    # Sort
    shap_imp_dict_d1 = dict(sorted(shap_imp_dict_d1.items(), key=lambda x: x[1], reverse=True))
    
    print("  Calculating SHAP values using TreeExplainer (Dataset 2)...")
    xgb_d2 = models_d2_all["XGBRegressor"]
    explainer_shap_d2 = shap.TreeExplainer(xgb_d2)
    # Explain all test samples (1322 points) instantaneously!
    shap_values_d2 = explainer_shap_d2.shap_values(X_test_2)
    
    # Calculate mean absolute SHAP value for feature importance
    shap_importance_d2 = np.mean(np.abs(shap_values_d2), axis=0)
    shap_imp_dict_d2 = {
        name: float(imp) for name, imp in zip(preproc_2['feature_cols'], shap_importance_d2)
    }
    # Sort
    shap_imp_dict_d2 = dict(sorted(shap_imp_dict_d2.items(), key=lambda x: x[1], reverse=True))
    
    # Save SHAP importances to JSON
    shap_results = {
        "dataset_1": shap_imp_dict_d1,
        "dataset_2": shap_imp_dict_d2
    }
    
    with open(os.path.join(MODELS_DIR, "shap_importance.json"), "w") as f:
        json.dump(shap_results, f, indent=4)
        
    print("  Saved SHAP feature importances.")
    print("SHAP explainability computation complete.")

if __name__ == "__main__":
    generate_shap_lime()
