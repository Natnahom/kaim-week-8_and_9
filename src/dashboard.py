# dashboard.py
import dash
from dash import dcc, html
import plotly.express as px
import requests
import pandas as pd

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),
    html.Div(id='summary-boxes'),
    dcc.Graph(id='fraud-trends'),
    dcc.Graph(id='device-fraud'),
    dcc.Graph(id='geographic-fraud')
])

# Callback to update summary boxes
@app.callback(
    dash.dependencies.Output('summary-boxes', 'children'),
    dash.dependencies.Input('summary-boxes', 'id')
)
def update_summary_boxes(_):
    response = requests.get('http://127.0.0.1:5000/api/summary').json()
    return [
        html.Div(f"Total Transactions: {response['total_transactions']}"),
        html.Div(f"Total Fraud Cases: {response['total_frauds']}"),
        html.Div(f"Fraud Percentage: {response['fraud_percentage']:.2f}%")
    ]

# Callback to update fraud trends graph
@app.callback(
    dash.dependencies.Output('fraud-trends', 'figure'),
    dash.dependencies.Input('fraud-trends', 'id')
)
def update_fraud_trends(_):
    response = requests.get('http://127.0.0.1:5000/api/fraud_trends').json()
    trend_data = pd.DataFrame(response)
    fig = px.line(trend_data, x='date', y='fraud_cases', title='Fraud Cases Over Time')
    return fig

# Callback to update device fraud graph
@app.callback(
    dash.dependencies.Output('device-fraud', 'figure'),
    dash.dependencies.Input('device-fraud', 'id')
)
def update_device_fraud(_):
    response = requests.get('http://127.0.0.1:5000/api/device_fraud').json()
    device_data = pd.DataFrame(response)
    fig = px.bar(device_data, x='device', y='fraud_cases', title='Fraud Cases by Device')
    return fig

# Callback to update geographic fraud graph
@app.callback(
    dash.dependencies.Output('geographic-fraud', 'figure'),
    dash.dependencies.Input('geographic-fraud', 'id')
)
def update_geographic_fraud(_):
    response = requests.get('http://127.0.0.1:5000/api/geographic_fraud').json()
    geo_data = pd.DataFrame(response)
    fig = px.bar(geo_data, x='location', y='fraud_cases', title='Fraud Cases by Location')
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)