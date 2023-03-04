import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.graph_objects as go

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.Div([
        html.Img(src='logo.png', className='logo'),
    ]),
    html.Div([
        dbc.Row([
            dbc.Col(
                dash.dcc.Graph(
                    figure=go.Figure(
                        data=[
                            go.Scatter(
                                x=[1, 2, 3],
                                y=[4, 5, 6],
                            )
                        ],
                        layout=go.Layout(
                            title='Bitcoin Chart'
                        )
                    )
                )
            )
        ]),
        dbc.Row([
            dbc.Col(
                dash.dcc.Graph(
                    figure=go.Figure(
                        data=[
                            go.Indicator(
                                mode='gauge+number',
                                value=50,
                                title='Gauge Meter'
                            )
                        ]
                    )
                )
            )
        ])
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
