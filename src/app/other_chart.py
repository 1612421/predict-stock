import dash_core_components as dcc
import dash_html_components as html
from pandas import DataFrame
from src.predict import StockManager

from dash.dash import Dash


def other_chart(app: Dash, df: DataFrame, stock_manager: StockManager):
    return html.Div(
        [
            html.H1("Stocks High vs Lows", style={'textAlign': 'center'}),
            dcc.Dropdown(
                id='my-dropdown',
                options=[
                    {
                        'label': 'Tesla',
                        'value': 'TSLA'
                    }, {
                        'label': 'Apple',
                        'value': 'AAPL'
                    }, {
                        'label': 'Facebook',
                        'value': 'FB'
                    }, {
                        'label': 'Microsoft',
                        'value': 'MSFT'
                    }
                ],
                multi=True,
                value=['FB'],
                style={
                    "display": "block",
                    "margin-left": "auto",
                    "margin-right": "auto",
                    "width": "60%"
                }
            ),
            dcc.Graph(id='highlow'),
            html.H1("Stocks Market Volume", style={'textAlign': 'center'}),
            dcc.Dropdown(
                id='my-dropdown2',
                options=[
                    {
                        'label': 'Tesla',
                        'value': 'TSLA'
                    }, {
                        'label': 'Apple',
                        'value': 'AAPL'
                    }, {
                        'label': 'Facebook',
                        'value': 'FB'
                    }, {
                        'label': 'Microsoft',
                        'value': 'MSFT'
                    }
                ],
                multi=True,
                value=['FB'],
                style={
                    "display": "block",
                    "margin-left": "auto",
                    "margin-right": "auto",
                    "width": "60%"
                }
            ),
            dcc.Graph(id='volume')
        ],
        className="container"
    )
