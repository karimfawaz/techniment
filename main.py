import dash_bootstrap_components as dbc
from dash import Dash, html

from frontend.app_layout import app_layout
from frontend.callbacks.app_callbacks import app_callbacks

app = Dash(__name__, title="Techniment",
           external_stylesheets=[dbc.themes.BOOTSTRAP])

html.Meta(charSet='UTF-8')
app.layout = app_layout(app)
app_callbacks(app)


if __name__ == "__main__":
    app.run_server(debug=True)
