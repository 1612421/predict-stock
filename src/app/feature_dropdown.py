import dash_html_components as html
import dash_core_components as dcc


def FeatureDropdownComponent(id: str):
    return html.Div(
        className='form-control',
        children=[
            html.Label('Choose data feature', htmlFor=id),
            dcc.Dropdown(
                id=id,
                options=[
                    {
                        'label': 'Close price only',
                        'value': 'Close'
                    }, {
                        'label': 'Rate of Change (on Close price)',
                        'value': 'ROC'
                    }
                ],
                value='Close',
                clearable=False,
                placeholder='Choose data feature'
            )
        ]
    )