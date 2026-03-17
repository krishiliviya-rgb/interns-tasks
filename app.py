import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Page setup
st.set_page_config(page_title="Sales Forecasting", page_icon="📊", layout="centered")

# Load banner image
image = Image.open("assets/banner.jpg")
st.image(image, use_container_width=True)

st.title("📊 Sales Demand Forecasting App")
st.markdown("### Predict Sales based on Stock, Price & Date")

st.markdown("---")

# Load trained model
model = joblib.load("models/sales_model.pkl")

# Load dataset for visualization
df = pd.read_csv("data/sales.csv")
df['data'] = pd.to_datetime(df['data'], errors='coerce')

st.subheader("📈 Sales vs Price")

fig, ax = plt.subplots()
ax.scatter(df['preco'], df['venda'])
ax.set_xlabel("Price")
ax.set_ylabel("Sales")
st.pyplot(fig)

st.markdown("---")

# Prediction Section
st.subheader("🔮 Predict Sales")

estoque = st.number_input("Enter Stock Quantity", min_value=0)
preco = st.number_input("Enter Price", min_value=0.0)

year = st.number_input("Enter Year", 2000, 2100)
month = st.number_input("Enter Month", 1, 12)
day = st.number_input("Enter Day", 1, 31)

if st.button("Predict Sales"):
    input_data = np.array([[estoque, preco, year, month, day]])
    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction)
    st.success(f"Predicted Sales: {prediction:.2f}")