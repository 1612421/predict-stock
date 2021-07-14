import dash_html_components as html
import dash_core_components as dcc


def PredictMethodDropdownComponent(id: str):
    return html.Div(
        className='form-control',
        children=[
            html.Label('Choose prediction method', htmlFor=id),
            dcc.Dropdown(
                id=id,
                options=[
                    {
                        'label': 'Long Short Time Memory (LSTM)',
                        'value': 'lstm'
                    }, {
                        'label': 'XGBoost',
                        'value': 'xgboost'
                    }
                ],
                value='lstm',
                searchable=True,
                clearable=False,
                placeholder='Choose prediction method'
            )
        ]
    )