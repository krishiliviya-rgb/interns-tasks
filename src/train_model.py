import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

# Load dataset
df = pd.read_csv("data/sales.csv")

# Convert date column to datetime
df['data'] = pd.to_datetime(df['data'], errors='coerce')

# Extract time features
df['year'] = df['data'].dt.year
df['month'] = df['data'].dt.month
df['day'] = df['data'].dt.day

# Drop rows with missing dates (if any)
df = df.dropna()

# Features and target
X = df[['estoque', 'preco', 'year', 'month', 'day']]
y = df['venda']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("📊 Model Evaluation:")
print(f"R2 Score: {r2:.2f}")
print(f"MAE: {mae:.2f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/sales_model.pkl")

print("✅ Model trained and saved successfully!")