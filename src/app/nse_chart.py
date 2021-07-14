import dash_core_components as dcc
import dash_html_components as html
from pandas import DataFrame
from plotly import graph_objects as go
from src.predict.StockManager import StockManager

from dash.dash import Dash


def nse_chart(
    app: Dash,
    df: DataFrame,
    stock_manager: StockManager,
    brand: str = 'AAPL',
    method: str = 'lstm'
):
    if (not stock_manager.is_loaded):
        return html.H3("Loading...")
    return html.Div(
        [
            html.H2("Actual closing price", style={"textAlign": "center"}),
            dcc.Graph(
                id="Actual Data",
                animate=True,
                figure={
                    "data":
                        [
                            go.Scatter(
                                x=df[df["Stock"] == brand].Date,
                                y=df[df["Stock"] == brand].Close,
                                mode='lines',
                                name='Real Close Price'
                            ),
                            go.Scatter(
                                x=stock_manager.predicted_chart[method][brand]
                                ['predict'].index,
                                y=stock_manager.predicted_chart[method][brand]
                                ['predict']['Close Predict'],
                                mode='lines',
                                name=f"Predict Close Price ({method})"
                            ),
                        ],
                    "layout":
                        go.Layout(
                            title='scatter plot',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'},
                            height=600
                        )
                }
            ),
            html.H2(
                "LSTM Predicted closing price", style={"textAlign": "center"}
            ),
            dcc.Graph(
                id="Predicted Data",
                figure={
                    "data":
                        [
                            go.Scatter(
                                x=stock_manager.lstm_nse_chart['predict'].index,
                                y=stock_manager.lstm_nse_chart['predict']
                                ['Close Predict'],
                                mode='lines',
                                name='Predict Close Price'
                            ),
                            go.Scatter(
                                x=stock_manager.xgboost_nse_chart['predict'].
                                index,
                                y=stock_manager.xgboost_nse_chart['predict']
                                ['Close Predict'],
                                mode='lines',
                                name='Predict Close Price (XGBoost)'
                            )
                        ],
                    "layout":
                        go.Layout(
                            title='scatter plot',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'}
                        )
                }
            )
        ]
    )
