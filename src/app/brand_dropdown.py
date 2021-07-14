import dash_html_components as html
import dash_core_components as dcc


def BrandDropdownComponent(id: str):
    return html.Div(
        className='form-control',
        children=[
            html.Label('Choose company', htmlFor=id),
            dcc.Dropdown(
                id=id,
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
                value='FB',
                searchable=True,
                clearable=False,
                placeholder='Choose company\'s data'
            )
        ]
    )