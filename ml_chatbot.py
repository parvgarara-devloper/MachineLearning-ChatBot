import os
import sys
import json
import ollama
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42

def call_llm(prompt):
    try:
        client = ollama.Client(host='http://127.0.0.1:11434', timeout=None)
        response = client.chat(
            model='qwen2.5',
            messages=[{'role': 'user', 'content': prompt}],
            stream=False
        )
        if hasattr(response, 'message'):
            return response.message.content
        return response.get('message', {}).get('content', "")
    except Exception as e:
        return None

def load_csv():
    while True:
        path = input("Enter CSV file path: ").strip()
        try:
            df = pd.read_csv(path)
            print("\n--- Dataset Loaded ---")
            print("Columns:", list(df.columns))
            print("\nData Types:\n", df.dtypes.to_string())
            print("\nNull Counts:\n", df.isnull().sum().to_string())
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}. Please try again.")

def analyze_dataset(df):
    print("\n--- Generating Dataset Profile ---")
    profile = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "unique_counts": df.nunique().to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "samples": {col: df[col].dropna().astype(str).head(5).tolist() for col in df.columns},
        "first_3_rows": df.head(3).astype(str).to_dict(orient="records")
    }
    
    prompt = f"Analyze this dataset profile:\n{json.dumps(profile, indent=2)}\n\nProvide:\n1. Dataset description\n2. Suggested target column\n3. Columns to drop\n4. Data quality issues"
    
    response = call_llm(prompt)
    if response:
        print("\n--- LLM Analysis ---")
        print(response)
    else:
        print("\n[OFFLINE] LLM unavailable for dataset analysis.")

def configure_pipeline(df):
    print("\n--- Interactive Configuration ---")
    while True:
        target_input = input("1. Enter target column name or index: ").strip()
        if target_input.isdigit():
            idx = int(target_input)
            if 0 <= idx < len(df.columns):
                target = df.columns[idx]
                break
        elif target_input in df.columns:
            target = target_input
            break
        print("Invalid column. Try again.")

    unique_target_vals = df[target].nunique()
    is_string = df[target].dtype == 'object'
    
    if unique_target_vals <= 20 or is_string:
        suggested_task = "classification"
    else:
        suggested_task = "regression"
        
    task_input = input(f"2. Enter task type (classification/regression) [Auto-suggested: {suggested_task}]: ").strip().lower()
    task = task_input if task_input in ["classification", "regression"] else suggested_task
    
    features_input = input("3. Feature selection (A: all except target, B: LLM suggestion (JSON list), C: manual selection) [A]: ").strip().upper()
    if features_input == 'C':
        manual_feats = input("Enter comma-separated features: ").strip()
        features = [f.strip() for f in manual_feats.split(',') if f.strip() in df.columns]
        if not features:
            print("No valid features selected. Defaulting to all except target.")
            features = [col for col in df.columns if col != target]
    elif features_input == 'B':
        llm_feats = input("Enter LLM suggestion (JSON list): ").strip()
        try:
            suggested = json.loads(llm_feats)
            features = [f for f in suggested if f in df.columns]
            if not features:
                raise ValueError("No valid features")
        except Exception:
            print("Invalid JSON or no valid features. Defaulting to all except target.")
            features = [col for col in df.columns if col != target]
    else:
        features = [col for col in df.columns if col != target]
        
    null_input = input("4. Null handling strategy (fill/drop) [fill]: ").strip().lower()
    null_strategy = null_input if null_input in ["fill", "drop"] else "fill"
    
    test_input = input("5. Test split ratio (0.05 - 0.50) [0.2]: ").strip()
    try:
        test_size = float(test_input)
        if not (0.05 <= test_size <= 0.50):
            test_size = 0.2
    except ValueError:
        test_size = 0.2
        
    return {
        "target": target,
        "task": task,
        "features": features,
        "null_strategy": null_strategy,
        "test_size": test_size
    }

def prepare_data(df, config):
    print("\n--- Data Preparation ---")
    data = df[config["features"] + [config["target"]]].copy()
    
    if config["null_strategy"] == "drop":
        data.dropna(inplace=True)
    else:
        for col in config["features"]:
            if data[col].dtype == 'object':
                mode_val = data[col].mode()
                if not mode_val.empty:
                    data[col] = data[col].fillna(mode_val[0])
            else:
                data[col] = data[col].fillna(data[col].median())
        if data[config["target"]].dtype == 'object':
            mode_val = data[config["target"]].mode()
            if not mode_val.empty:
                data[config["target"]] = data[config["target"]].fillna(mode_val[0])
        else:
            data[config["target"]] = data[config["target"]].fillna(data[config["target"]].median())
            
    X = data[config["features"]]
    y = data[config["target"]]
    
    encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X.loc[:, col] = X[col].astype(str)
            le.fit(X[col])
            X.loc[:, col] = le.transform(X[col])
            encoders[col] = le
            
    target_encoder = None
    if config["task"] == "classification":
        if y.dtype == 'object' or str(y.dtype) == 'category':
            target_encoder = LabelEncoder()
            y = y.astype(str)
            y = target_encoder.fit_transform(y)
        else:
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
            
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_size"], random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test, encoders, target_encoder

def train_model(X_train, y_train, config):
    print("\n--- Model Training ---")
    est_input = input("Enter n_estimators [100]: ").strip()
    try:
        n_estimators = int(est_input)
    except ValueError:
        n_estimators = 100
        
    if config["task"] == "classification":
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=1)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE, n_jobs=1)
        
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, config):
    print("\n--- Model Evaluation ---")
    predictions = model.predict(X_test)
    metrics = {}
    
    if config["task"] == "classification":
        acc = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=False)
        report_dict = classification_report(y_test, predictions, output_dict=True)
        metrics = {"accuracy": acc, "report": report_dict}
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:\n", report)
    else:
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        metrics = {"MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}")
        
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    top_features = [(config["features"][i], importances[i]) for i in indices]
    
    print("\nTop 10 Feature Importances:")
    for feat, imp in top_features:
        bar_len = int(imp * 50)
        bar = '█' * bar_len
        print(f"{feat[:15]:<15} | {imp:.4f} | {bar}")
        
    return metrics, top_features

def explain_model(config, metrics, top_features):
    print("\n--- LLM Explanation ---")
    prompt = f"Explain these modeling results:\nTask: {config['task']}\nTarget: {config['target']}\nMetrics: {json.dumps(metrics, indent=2)}\nTop Features: {json.dumps(top_features, indent=2)}\n\nProvide:\n1. What the model learned\n2. Why top features matter\n3. Meaning of metrics\n4. Improvement suggestions"
    
    response = call_llm(prompt)
    if response:
        print(response)
    else:
        print("[OFFLINE] LLM unavailable for model explanation.")
    return response

def predict_sample(model, config, encoders, target_encoder):
    while True:
        do_pred = input("\nMake a prediction on a new sample? (y/N): ").strip().lower()
        if do_pred != 'y':
            break
            
        print("Enter feature values:")
        sample = {}
        for col in config["features"]:
            val = input(f"{col}: ").strip()
            if col in encoders:
                try:
                    if val in encoders[col].classes_:
                        sample[col] = encoders[col].transform([val])[0]
                    else:
                        sample[col] = 0
                except Exception:
                    sample[col] = 0
            else:
                try:
                    sample[col] = float(val)
                except ValueError:
                    sample[col] = 0.0
                    
        sample_df = pd.DataFrame([sample])
        pred = model.predict(sample_df)[0]
        
        if config["task"] == "classification":
            if target_encoder:
                pred_label = target_encoder.inverse_transform([int(pred)])[0]
            else:
                pred_label = pred
            print(f"\nPrediction: {pred_label}")
            
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(sample_df)[0]
                top_3_idx = np.argsort(probs)[::-1][:3]
                print("Top 3 Probabilities:")
                for idx in top_3_idx:
                    lbl = target_encoder.inverse_transform([idx])[0] if target_encoder else idx
                    print(f"  {lbl}: {probs[idx]:.4f}")
        else:
            print(f"\nPrediction: {pred:.4f}")

def save_report(config, metrics, top_features, llm_explanation):
    do_save = input("\nSave results to report? (y/N): ").strip().lower()
    if do_save == 'y':
        path = input("Enter file path [ml_results.txt]: ").strip()
        if not path:
            path = "ml_results.txt"
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write("=== ML Analyst Report ===\n\n")
                f.write("Configuration:\n")
                f.write(json.dumps(config, indent=2) + "\n\n")
                f.write("Metrics:\n")
                f.write(json.dumps(metrics, indent=2) + "\n\n")
                f.write("Top Features:\n")
                for feat, imp in top_features:
                    f.write(f"{feat}: {imp:.4f}\n")
                f.write("\nLLM Explanation:\n")
                f.write(llm_explanation if llm_explanation else "[OFFLINE]")
            print(f"Report saved to {path}")
        except Exception as e:
            print(f"Error saving report: {e}")

def main():
    print("Welcome to ML Analyst (LLM-Assisted AutoML CLI Tool)")
    df = load_csv()
    analyze_dataset(df)
    config = configure_pipeline(df)
    X_train, X_test, y_train, y_test, encoders, target_encoder = prepare_data(df, config)
    model = train_model(X_train, y_train, config)
    metrics, top_features = evaluate_model(model, X_test, y_test, config)
    llm_explanation = explain_model(config, metrics, top_features)
    predict_sample(model, config, encoders, target_encoder)
    save_report(config, metrics, top_features, llm_explanation)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    pd.options.mode.chained_assignment = None
    main()
