import dash_bootstrap_components as dbc
from dash import html

from frontend.components.feargreed import feargreed

from .components.chart import chart


def app_layout(app):
    return dbc.Container(children=[
        dbc.Navbar(

            dbc.Row(
                [
                    dbc.Col(
                        html.Img(src="assets/logo.png", height="50px")),
                    dbc.Col(dbc.NavbarBrand(
                        "Techniment", className="ms-2")),
                ],
                align="center",

                className="g-0",

            ),

            color="dark",
            dark=True,
            class_name="navbar"

        ),

        dbc.Row(className='', children=[


            dbc.Col([

                html.Div(id='', className='', children=chart()),
            ], lg='8', md='8', sm='8'),
            dbc.Col([
                html.Div(id='', className='', children=feargreed()),
            ], lg='4', md='4', sm='4'),
        ], align="center"),

    ], class_name="page", fluid=True)
