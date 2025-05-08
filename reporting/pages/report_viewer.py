import json
import os
from dash import html, dcc, callback, Input, Output, State, PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go # Plotlyグラフオブジェクト用

# グラフ描画コンポーネントをインポート
from reporting.components import charts

# レポートデータをロードする関数 (仮)
def load_report_data(slug):
    path = os.path.join('outputs', slug, 'hierarchical_result.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading report data for {slug}: {e}")
        return None

# レポート詳細ページのレイアウト定義
def layout(slug=None):
    if slug is None:
        return html.Div("レポートIDが指定されていません。")

    report_data = load_report_data(slug)

    if report_data is None:
        return html.Div(f"レポート '{slug}' が見つかりません。")

    # dcc.Storeコンポーネントを追加してレポートデータを保持
    return dbc.Container([
        dcc.Store(id='report-data-store', data=report_data),
        dcc.Store(id='chart-config-store', data={ # チャート設定用Store
            'selected_chart': 'scatterAll',
            'max_density': 0.2,
            'min_value': 5,
            'show_labels': True,
            'treemap_level': '0',
            'is_dense_group_enabled': True # 初期状態
        }),
        html.H2(report_data.get("config", {}).get("question", "レポート詳細")),
        html.P(report_data.get("overview", "概要はありません。")),

        # --- 可視化コントロール ---
        dbc.Row([
            dbc.Col(dcc.RadioItems(
                id='chart-type-selector',
                options=[
                    {'label': '全体図 (Scatter All)', 'value': 'scatterAll'},
                    {'label': '濃い意見グループ (Scatter Dense)', 'value': 'scatterDensity'},
                    {'label': '階層図 (Treemap)', 'value': 'treemap'},
                ],
                value='scatterAll', # 初期値
                inline=True,
                className="mb-2"
            ), width=12),
        ], className="mb-3"),

        # 密度フィルターなどの設定項目（最初は非表示、Dense選択時に表示）
        dbc.Collapse(
            dbc.Card(dbc.CardBody([
                 html.Div([
                     html.Label("上位何％の意見グループを表示するか (%)"),
                     dcc.Slider(id='density-slider', min=5, max=100, step=5, value=20, marks={i: f'{i}%' for i in range(5, 101, 15)}),
                 ], className="mb-3"),
                 html.Div([
                     html.Label("意見グループのサンプル数の最小数"),
                     dcc.Slider(id='min-value-slider', min=1, max=10, step=1, value=5, marks={i: f'{i}' for i in range(1, 11)}),
                 ]),
                 dbc.Checkbox(id='show-labels-checkbox', label="クラスターラベルを表示", value=True, className="mt-3"),
            ])),
            id="dense-settings-collapse",
            is_open=False, # 初期状態は閉じる
        ),

        # Treemap用操作ボタン（Treemap選択時に表示）
         dbc.Collapse(
            dbc.Button("親階層に戻る", id="treemap-back-button", color="secondary", size="sm", className="mb-3", disabled=True),
            id="treemap-controls-collapse",
            is_open=False,
        ),

        # グラフ表示エリア
        dbc.Spinner(html.Div(id='graph-output', style={'height': '700px'})),

        # TODO: 選択されたクラスターの詳細表示エリア
        html.Div(id='cluster-detail-output'),

    ], fluid=True, className="mt-4")

# コールバックはこのファイルではなく、callbacks/report_viewer_callbacks.py に記述することを推奨