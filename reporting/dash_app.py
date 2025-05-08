import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# components と pages をインポート
from reporting.components import header, footer
from reporting.pages import report_list, report_viewer, report_creator

# Dashアプリケーションの初期化
# スタイルシートはdbc.themes.BOOTSTRAPやassets/custom.cssなどを指定
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server # gunicorn等でデプロイする場合に必要

# アプリケーション全体のレイアウト
app.layout = html.Div([
    dcc.Location(id='url', refresh=False), # URLの変化を検知
    header.layout(), # ヘッダーコンポーネント
    dbc.Container(id='page-content', fluid=True), # ページコンテンツ表示エリア
    footer.layout() # フッターコンポーネント
])

# URLに基づいてページコンテンツを切り替えるコールバック
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/create':
        return report_creator.layout()
    elif pathname is not None and pathname.startswith('/report/'):
        # '/report/your-report-slug' のようなURLからslugを取得
        slug = pathname.split('/report/')[-1]
        return report_viewer.layout(slug)
    else:
        # デフォルトはレポート一覧ページ
        return report_list.layout()

# --- 他のページ間共通のコールバックや、各ページからimportしたコールバックを登録 ---
from reporting.callbacks import report_creator_callbacks, report_viewer_callbacks
report_creator_callbacks.register_callbacks(app)
report_viewer_callbacks.register_callbacks(app)
# --- ここまで ---

if __name__ == '__main__':
    # 開発用にDashサーバーを起動
    # TODO: register_callbacks を呼び出す
    from reporting.callbacks import report_creator_callbacks, report_viewer_callbacks
    report_creator_callbacks.register_callbacks(app)
    report_viewer_callbacks.register_callbacks(app)
    app.run_server(debug=True, host='0.0.0.0', port=8050)