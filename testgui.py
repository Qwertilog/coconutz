# train_qda_all.py
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import joblib
from pathlib import Path

# --- File paths ---
base_dir = Path(__file__).parent
csv_path = base_dir / "coconut_features.csv"
model_path = base_dir / "camera_model.pkl"

# --- Load data ---
print("Loading dataset from:", csv_path)
df = pd.read_csv(csv_path)

if not {'Label', 'H_mean', 'S_mean', 'V_mean'}.issubset(df.columns):
    raise ValueError("CSV must contain columns: Label, H_mean, S_mean, V_mean")

# --- Prepare features and labels ---
X = df[['H_mean', 'S_mean', 'V_mean']].values
y = df['Label'].values

# --- Show dataset summary ---
print("\n=== Dataset Summary ===")
print("Total samples:", len(df))
print(df['Label'].value_counts())

# --- Train QDA model ---
print("\nTraining Quadratic Discriminant Analysis model on ALL data...")
qda = QuadraticDiscriminantAnalysis()
qda.fit(X, y)
print("✅ Training complete!")

# --- Save trained model ---
joblib.dump(qda, model_path)
print(f"Model saved as: {model_path}")

# --- Optional check ---
example = X[0].reshape(1, -1)
pred = qda.predict(example)[0]
print(f"\nExample prediction for first sample: {pred}")
