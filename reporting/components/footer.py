from dash import html
import dash_bootstrap_components as dbc

def layout():
    # TODO: フッター情報を充実させる
    return html.Footer(
        dbc.Container("© 2024 Your Name or Organization", className="text-center text-muted py-3"),
        className="mt-5"
    )