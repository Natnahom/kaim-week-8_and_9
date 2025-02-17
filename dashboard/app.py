from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

# Load the fraud data
data = pd.read_csv('../../Data/Fraud_Data.csv')  
data['purchase_time'] = pd.to_datetime(data['purchase_time'])

@app.route('/api/summary')
def summary():
    total_transactions = len(data)
    fraud_cases = data[data['class'] == 1].shape[0]
    fraud_percentage = (fraud_cases / total_transactions) * 100
    return jsonify({
        'total_transactions': total_transactions,
        'fraud_cases': fraud_cases,
        'fraud_percentage': round(fraud_percentage, 2)
    })

@app.route('/api/trends')
def fraud_trends():
    trend_data = data.groupby(data['purchase_time'].dt.date)['class'].sum().reset_index()
    trend_data.columns = ['date', 'fraud_cases']
    return jsonify(trend_data.to_dict(orient='records'))

@app.route('/api/device_browser')
def device_browser():
    device_browser_data = data.groupby(['device_id', 'browser'])['class'].sum().reset_index()
    # device_browser_data.columns = ['device_id', 'browser', 'fraud_cases']
    return jsonify(device_browser_data.to_dict(orient='records'))

@app.route('/api/age')
def age_fraud():
    geo_data = data.groupby('age')['class'].sum().reset_index()
    geo_data.columns = ['age', 'fraud_cases']
    return jsonify(geo_data.to_dict(orient='records'))

@app.route('/api/purchase_value')
def purchase_fraud():
    geo_data = data.groupby('purchase_value')['class'].sum().reset_index()
    geo_data.columns = ['purchase_value', 'fraud_cases']
    return jsonify(geo_data.to_dict(orient='records'))

@app.route('/api/geographic')
def geographic_fraud():
    geo_data = data.groupby('ip_address')['class'].sum().reset_index()
    geo_data.columns = ['location', 'fraud_cases']
    return jsonify(geo_data.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)