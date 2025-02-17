from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import requests
import pandas as pd

app = Dash(__name__, external_stylesheets=['./static/style.css'])

app.layout = html.Div(children=[
    html.H1("Fraud Detection Dashboard", style={'textAlign': 'center', 'color': '#3377FF'}),
    
    html.Div(id='summary-boxes', className='row', style={'display': 'flex', 'justifyContent': 'space-around'}),
    
    dcc.Graph(id='fraud-trends'),
    
    dcc.Graph(id='purchaseValue-fraud'),

    dcc.Graph(id='age-fraud'),
    
    dcc.Graph(id='geographic-fraud'),
    
    dcc.Graph(id='device-browser-fraud')
])

@app.callback(
    Output('summary-boxes', 'children'),
    Input('summary-boxes', 'id')
)
def update_summary_boxes(_):
    response = requests.get('http://127.0.0.1:5000/api/summary')
    summary_data = response.json()
    return [
        html.Div([
            html.H3("Total Transactions"),
            html.P(f"{summary_data['total_transactions']}")
        ], className='summary-box'),
        
        html.Div([
            html.H3("Fraud Cases"),
            html.P(f"{summary_data['fraud_cases']}", style={'color': '#FF2211'})
        ], className='summary-box'),
        
        html.Div([
            html.H3("Fraud Percentage"),
            html.P(f"{summary_data['fraud_percentage']}%", style={'color': '#FF2211'})
        ], className='summary-box')
    ]

@app.callback(
    Output('fraud-trends', 'figure'),
    Input('fraud-trends', 'id')
)
def update_fraud_trends(_):
    response = requests.get('http://127.0.0.1:5000/api/trends')
    trends_data = response.json()
    df_trends = pd.DataFrame(trends_data)
    fig = px.line(df_trends, x='date', y='fraud_cases', title='Fraud Cases Over Time')
    return fig

@app.callback(
    Output('purchaseValue-fraud', 'figure'),
    Input('purchaseValue-fraud', 'id')
)
def update_purchase_fraud(_):
    response = requests.get('http://127.0.0.1:5000/api/purchase_value')
    geo_data = response.json()
    df_geo = pd.DataFrame(geo_data)
    fig = px.bar(df_geo, x='purchase_value', y='fraud_cases', title='Fraud Cases by Purchase Value')
    return fig

@app.callback(
    Output('age-fraud', 'figure'),
    Input('age-fraud', 'id')
)
def update_age_fraud(_):
    response = requests.get('http://127.0.0.1:5000/api/age')
    geo_data = response.json()
    df_geo = pd.DataFrame(geo_data)
    fig = px.bar(df_geo, x='age', y='fraud_cases', title='Fraud Cases by Age')
    return fig

@app.callback(
    Output('geographic-fraud', 'figure'),
    Input('geographic-fraud', 'id')
)
def update_geographic_fraud(_):
    response = requests.get('http://127.0.0.1:5000/api/geographic')
    geo_data = response.json()
    df_geo = pd.DataFrame(geo_data)
    fig = px.bar(df_geo, x='location', y='fraud_cases', title='Fraud Cases by Location')
    return fig

@app.callback(
    Output('device-browser-fraud', 'figure'),
    Input('device-browser-fraud', 'id')
)
def update_device_browser_fraud(_):
    response = requests.get('http://127.0.0.1:5000/api/device_browser')
    device_browser_data = response.json()
    df_device_browser = pd.DataFrame(device_browser_data)
    fig = px.bar(df_device_browser, x='browser', y='fraud_cases', color='device_id', title='Fraud Cases by Device and Browser')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)