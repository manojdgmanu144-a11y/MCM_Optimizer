"""
src/model.py
------------
Step 2: Train a Linear Regression model, evaluate it, and save it for the web app.
Run this after data_loader: python src/model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

def train_and_save_model():
    # 1. Load the data prepared in Step 1
    if not os.path.exists("data/student-mat.csv"):
        print("Error: data/student-mat.csv not found. Run data_loader.py first!")
        return

    df = pd.read_csv("data/student-mat.csv", sep=";")
    print(f"✓ Loaded {len(df)} records for training.")

    # 2. Feature Selection
    # We'll use the strongest predictors found during EDA
    features = ['studytime', 'failures', 'absences', 'G1', 'G2']
    target = 'G3'
    
    X = df[features]
    y = df[target]

    # 3. Split into Training and Testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Train the Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n" + "=" * 30)
    print("MODEL PERFORMANCE")
    print("=" * 30)
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score (Accuracy): {r2:.2f}")
    print("=" * 30)

    # 6. Save the Model
    os.makedirs("models", exist_ok=True)
    with open("models/student_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print("\n✓ Model saved successfully to: models/student_model.pkl")
    print("Next Step: Run your Streamlit app to use the model!")

if __name__ == "__main__":
    train_and_save_model()