from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

# Load the fraud data
data = pd.read_csv('../../Data/Fraud_Data.csv')  

# Convert purchase_time to datetime
data['purchase_time'] = pd.to_datetime(data['purchase_time'])

@app.route('/api/summary', methods=['GET'])
def summary():
    total_transactions = len(data)
    total_frauds = data[data['class'] == 1].shape[0]  # 'class' column indicates fraud
    fraud_percentage = (total_frauds / total_transactions) * 100 if total_transactions > 0 else 0
    
    return jsonify({
        'total_transactions': total_transactions,
        'total_frauds': total_frauds,
        'fraud_percentage': fraud_percentage
    })

@app.route('/api/fraud_trends', methods=['GET'])
def fraud_trends():
    # Use purchase_time for trends
    trend_data = data.groupby(data['purchase_time'].dt.date)['class'].sum().reset_index()
    trend_data.columns = ['date', 'fraud_cases']  # Renaming columns for clarity
    return trend_data.to_json(orient='records')

@app.route('/api/device_fraud', methods=['GET'])
def device_fraud():
    device_data = data.groupby('device_id')['class'].sum().reset_index()
    device_data.columns = ['device', 'fraud_cases']  # Renaming columns for clarity
    return device_data.to_json(orient='records')

@app.route('/api/geographic_fraud', methods=['GET'])
def geographic_fraud():
    geo_data = data.groupby('ip_address')['class'].sum().reset_index()
    geo_data.columns = ['location', 'fraud_cases']  # Renaming columns for clarity
    return geo_data.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)