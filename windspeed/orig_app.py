import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px
# Load the model
model = joblib.load('wind_speed_model12.pkl')

# Initialize the scaler
scaler = StandardScaler()

# Predefined column names
columns = ['year', 'month', 'day', 'hour',  'avg_speed'	,'s50',	's20'	,'pressure',	'w98mWV',	'w78mWV',	'w48mWV',	'temperature',	'humidity']	  # Replace with actual feature names

st.title("Wind Speed Prediction")

# Input fields for user to enter data
year = st.number_input("Year", min_value=2000, max_value=2100, value=2021)
month = st.number_input("Month", min_value=1, max_value=12, value=1)
day = st.number_input("Day", min_value=1, max_value=31, value=1)
hour = st.number_input("Hour", min_value=0, max_value=23, value=0)
avg_speed= st.number_input("avg_speed")
s50= st.number_input("s50")
s20= st.number_input("s20")
pressure= st.number_input("pressure")
w98mWV= st.number_input("w98mWV")
w78mWV= st.number_input("w78mWV")
w48mWV= st.number_input("w48mWV")
temperature= st.number_input("temperature")
humidity= st.number_input("humidity")

# Create a dictionary to store the input data
input_data = {
    'year': year,'month': month,'day': day,'hour': hour,'avg_speed': avg_speed,'s50': s50,'s20': s20,'pressure': pressure,'w98mWV': w98mWV,
    'w78mWV': w78mWV,'w48mWV': w48mWV,'temperature': temperature,'humidity': humidity
}
# Convert the input data to a DataFrame
input_df = pd.DataFrame([input_data])
# Fit the scaler on the training data
scaler.fit(input_df)
# Standardize the input data
input_df_scaled = scaler.transform(input_df)
# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df_scaled)
    st.write(f"Predicted Wind Speed at 80m is: {prediction[0]}")
    # Feature importance visualization
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        features_df = pd.DataFrame({
            'Feature': columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # Plotly bar plot for feature importances
        fig = px.bar(features_df, x='Feature', y='Importance', title='Feature Importances')
        st.plotly_chart(fig)
    # Input data visualization
    fig, ax = plt.subplots()
    input_df.plot(kind='bar', ax=ax)
    plt.title('Input Data Visualization')
    plt.xlabel('Features')
    plt.ylabel('Values')
    st.pyplot(fig)