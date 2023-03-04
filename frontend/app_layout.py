import dash_bootstrap_components as dbc
from dash import html

from frontend.components.feargreed import feargreed

from .components.chart import chart


def app_layout(app):
    return dbc.Container([

        html.Img(id='', className='', children=[],
                 src='assets/logo.png', alt=''),

        html.Div(id='', className='', children=chart()),
        html.Div(id='', className='', children=feargreed()),

    ]
    )
