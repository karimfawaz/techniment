from dash import Dash, Input, Output, dcc, html


def chart():
    return html.Div([

        dcc.Checklist(
            id='toggle-rangeslider',
            options=[{'label': 'Include Rangeslider',
                      'value': 'slider'}],
            value=['slider']
        ),
        dcc.Graph(id="graph"),
    ])
