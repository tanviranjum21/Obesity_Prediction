import requests
import json
import pandas as pd

# Define the URL of the Flask API endpoint
url = 'http://127.0.0.1:5000/predict'  # Change the URL if needed

# Read sample data from CSV file
sample_data_csv = '/Users/tanvir/Desktop/ML_Project/Obesity_Prediction_Project/Dataset/test.csv'
sample_data_df = pd.read_csv(sample_data_csv)

# Convert sample data DataFrame to JSON format
sample_data_json = sample_data_df.to_json(orient='records')

# Make a POST request to the Flask API endpoint
response = requests.post(url, json=sample_data_json)

# Check if the request was successful
if response.status_code == 200:
    # Get the prediction result from the response
    prediction_result = response.json()
    print("Prediction Result:", prediction_result)
else:
    print("Failed to make prediction. Status code:", response.status_code)