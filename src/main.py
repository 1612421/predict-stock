from os import path
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash_html_components.Div import Div
import pandas as pd
from plotly import graph_objects as go
from dash.dependencies import Input, Output

from src.app import nse_chart, other_chart, BrandDropdownComponent, PredictMethodDropdownComponent
from src.config.file import PREDICT_DATA_FILE, TRAIN_FILE
from src.predict.ParseDF import parseCloseDF
from src.predict.StockManager import StockManager

stock_manager = StockManager()
df = pd.read_csv(PREDICT_DATA_FILE)
temp = parseCloseDF(TRAIN_FILE)
# apple_data = df[df['Stock'] == 'AAPL']
# new_df = apple_data.filter(['Close'])
# last_60_days = new_df[-60:].values
# predict_lstm.predict(last_60_days)
# print(stock_manager.close_chart['real'].index)

app = dash.Dash(
    assets_folder=path.abspath(f"{path.dirname(__file__)}/app/styles"),
    title="Predict Stock price"
)
server = app.server

app.layout = html.Div(
    [
        html.H1(
            "Stock Price Analysis Dashboard", style={"textAlign": "center"}
        ),
        dcc.Input('load', style={'display': 'none'}),
        html.Div(
            className='two-pane-view',
            children=[
                html.Div(
                    className="sidebar",
                    children=[
                        BrandDropdownComponent(id='brand-dropdown'),
                        PredictMethodDropdownComponent(id='method-dropdown')
                    ]
                ),
                html.Div(
                    className='content',
                    children=[
                        dcc.Tabs(
                            id="tabs",
                            children=[
                                dcc.Tab(
                                    id='nse_chart',
                                    label='NSE-TATAGLOBAL Stock Data',
                                    children=[
                                        nse_chart(
                                            app=app,
                                            df=df,
                                            stock_manager=stock_manager
                                        )
                                    ]
                                ),
                                dcc.Tab(
                                    label='Other Stock Data',
                                    children=[
                                        other_chart(app, df, stock_manager)
                                    ]
                                )
                            ]
                        )
                    ]
                ),
            ]
        ),
    ]
)


@app.callback(Output('highlow', 'figure'), [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {
        "TSLA": "Tesla",
        "AAPL": "Apple",
        "FB": "Facebook",
        "MSFT": "Microsoft",
    }
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
            go.Scatter(
                x=df[df["Stock"] == stock]["Date"],
                y=df[df["Stock"] == stock]["High"],
                mode='lines',
                opacity=0.7,
                name=f'High {dropdown[stock]}',
                textposition='bottom center'
            )
        )
        trace2.append(
            go.Scatter(
                x=df[df["Stock"] == stock]["Date"],
                y=df[df["Stock"] == stock]["Low"],
                mode='lines',
                opacity=0.6,
                name=f'Low {dropdown[stock]}',
                textposition='bottom center'
            )
        )
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {
        'data':
            data,
        'layout':
            go.Layout(
                colorway=[
                    "#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400',
                    '#FF0056'
                ],
                height=600,
                title=
                f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
                xaxis={
                    "title": "Date",
                    'rangeselector':
                        {
                            'buttons':
                                list(
                                    [
                                        {
                                            'count': 1,
                                            'label': '1M',
                                            'step': 'month',
                                            'stepmode': 'backward'
                                        }, {
                                            'count': 6,
                                            'label': '6M',
                                            'step': 'month',
                                            'stepmode': 'backward'
                                        }, {
                                            'step': 'all'
                                        }
                                    ]
                                )
                        },
                    'rangeslider': {
                        'visible': True
                    },
                    'type': 'date'
                },
                yaxis={"title": "Price (USD)"}
            )
    }
    return figure


@app.callback(Output('volume', 'figure'), [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {
        "TSLA": "Tesla",
        "AAPL": "Apple",
        "FB": "Facebook",
        "MSFT": "Microsoft",
    }
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
            go.Scatter(
                x=df[df["Stock"] == stock]["Date"],
                y=df[df["Stock"] == stock]["Volume"],
                mode='lines',
                opacity=0.7,
                name=f'Volume {dropdown[stock]}',
                textposition='bottom center'
            )
        )
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {
        'data':
            data,
        'layout':
            go.Layout(
                colorway=[
                    "#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400',
                    '#FF0056'
                ],
                height=600,
                title=
                f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
                xaxis={
                    "title": "Date",
                    'rangeselector':
                        {
                            'buttons':
                                list(
                                    [
                                        {
                                            'count': 1,
                                            'label': '1M',
                                            'step': 'month',
                                            'stepmode': 'backward'
                                        }, {
                                            'count': 6,
                                            'label': '6M',
                                            'step': 'month',
                                            'stepmode': 'backward'
                                        }, {
                                            'step': 'all'
                                        }
                                    ]
                                )
                        },
                    'rangeslider': {
                        'visible': True
                    },
                    'type': 'date'
                },
                yaxis={"title": "Transactions Volume"}
            )
    }
    return figure


@app.callback(
    Output('nse_chart', component_property='children'),
    Input('brand-dropdown', component_property='value'),
    Input('method-dropdown', component_property='value')
)
def on_load(brand_value: str, method_value: str):
    if (not stock_manager.is_loaded):
        stock_manager.load_all()
    return [nse_chart(app, df, stock_manager, brand_value, method=method_value)]


if __name__ == '__main__':
    # prepare data before serve
    stock_manager.load_all()

    app.run_server(debug=True)
