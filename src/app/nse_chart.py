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
    method: str = 'lstm',
    feature: str = 'Close'
):
    if (not stock_manager.is_loaded):
        return html.H3("Loading...")
    data_by_brand = df[df["Stock"] == brand]
    dateList = stock_manager.predicted_future_chart[method][brand].index
    predict_train_values = stock_manager.predicted_chart[method][brand]
    predict_future_values = stock_manager.predicted_future_chart[method][brand]
    predict_data_column = f"{feature} Predict"

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
                                x=data_by_brand.Date,
                                y=data_by_brand[feature],
                                mode='lines',
                                name=f'Real {feature} price'
                            ),
                            go.Scatter(
                                x=predict_train_values['predict'].index,
                                y=predict_train_values['predict']
                                [predict_data_column],
                                mode='lines+markers',
                                name=f"Predict Train Price ({method})"
                            ),
                            go.Scatter(
                                x=predict_future_values.index,
                                y=predict_future_values[predict_data_column],
                                mode='lines+markers',
                                name=f"Predict Next Week Price ({method})"
                            ),
                        ],
                    "layout":
                        go.Layout(
                            title='scatter plot',
                            xaxis={'title': 'Date'},
                            yaxis={'title': 'Closing Rate'},
                            height=600,
                            xaxis_range=[dateList[-60], dateList[-1]],
                            yaxis_range=[
                                predict_future_values[predict_data_column] - 50,
                                predict_future_values[predict_data_column] + 50,
                            ]
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
