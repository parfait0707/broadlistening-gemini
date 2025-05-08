from dash import html
import dash_bootstrap_components as dbc

def layout():
    # TODO: 必要に応じてロゴやナビゲーションを追加
    return dbc.NavbarSimple(
        brand="階層的意見分析レポート",
        brand_href="/",
        color="primary",
        dark=True,
        className="mb-4"
    )