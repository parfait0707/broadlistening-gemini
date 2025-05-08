from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc

# プロンプトのデフォルト値を取得
from server.broadlistening.pipeline.prompts.extraction.default import extraction_prompt
from server.broadlistening.pipeline.prompts.hierarchical_initial_labelling.default import initial_labelling_prompt
from server.broadlistening.pipeline.prompts.hierarchical_merge_labelling.default import merge_labelling_prompt
from server.broadlistening.pipeline.prompts.hierarchical_overview.default import overview_prompt

def layout():
    return dbc.Container([
        html.H2("新しいレポートを作成", className="mb-4"),

        # 基本情報
        dbc.Card(dbc.CardBody([
            html.H4("基本情報", className="card-title mb-3"),
            dbc.Row([
                dbc.Col(dbc.Label("タイトル (問い)"), width=3),
                dbc.Col(dcc.Input(id='report-question', type='text', placeholder="例: 日本の未来について", className="form-control"), width=9),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(dbc.Label("調査概要 (イントロ)"), width=3),
                dbc.Col(dcc.Textarea(id='report-intro', placeholder="例: このレポートは...", className="form-control", rows=3), width=9),
            ], className="mb-3"),
             dbc.Row([
                dbc.Col(dbc.Label("レポートID (半角英数字とハイフン)"), width=3),
                dbc.Col(dcc.Input(id='report-slug', type='text', placeholder="例: japan-future-2024", className="form-control"), width=9),
            ], className="mb-3"),
             dbc.Alert(id='slug-validation-alert', color="danger", is_open=False),
        ]), className="mb-4"),

        # データ入力
        dbc.Card(dbc.CardBody([
             html.H4("入力データ", className="card-title mb-3"),
             dcc.RadioItems(
                id='input-type-selector',
                options=[
                    {'label': 'CSVファイルをアップロード', 'value': 'file'},
                    {'label': 'Google Spreadsheet URL', 'value': 'spreadsheet'},
                ],
                value='file',
                inline=True,
                className="mb-3"
             ),
             # ファイルアップロード用
             html.Div([
                 dcc.Upload(
                    id='upload-csv',
                    children=html.Div(['ファイルをドラッグ＆ドロップ または ', html.A('ファイルを選択')]),
                    style={ # 簡単なスタイル例
                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '1px', 'borderStyle': 'dashed',
                        'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
                    },
                    multiple=False # 単一ファイルのみ
                 ),
                 html.Div(id='output-upload-state'), # アップロード状態表示用
             ], id='file-upload-div'),
             # Spreadsheet用
             html.Div([
                 dcc.Input(id='spreadsheet-url', type='text', placeholder="https://docs.google.com/spreadsheets/d/...", className="form-control mb-2"),
                 dbc.Button("データを取得して確認", id='fetch-spreadsheet-button', color="info", size="sm"),
                 html.Div(id='spreadsheet-load-state'), # 読み込み状態表示用
             ], id='spreadsheet-div', style={'display': 'none'}), # 初期状態は非表示

             # カラム選択 (データ読み込み後に表示)
             html.Div([
                 dbc.Label("コメントが含まれるカラムを選択:", className="mt-3"),
                 dcc.Dropdown(id='comment-column-selector', options=[], placeholder="カラムを選択...", clearable=False),
             ], id='column-selection-div', style={'display': 'none'}),

        ]), className="mb-4"),

        # クラスタリング設定
        dbc.Card(dbc.CardBody([
            html.H4("階層クラスタリング設定", className="card-title mb-3"),
            dbc.Label("生成する階層数と各階層のクラスタ数をカンマ区切りで入力 (例: 3, 6, 12):"),
            dcc.Input(id='cluster-levels-input', type='text', value='3, 6, 12, 24', className="form-control"),
            # TODO: コメント数に応じた推奨値表示エリア (コールバックで更新)
        ]), className="mb-4"),

        # AI設定 (アコーディオンなどで隠しても良い)
        dbc.Accordion([
            dbc.AccordionItem([
                dbc.AccordionHeader("AI詳細設定 (オプション)", target_id="ai-settings-collapse"),
                dbc.AccordionBody(accordion_item_id="ai-settings-collapse", children=[
                     dbc.Row([
                         dbc.Col(dbc.Label("LLMプロバイダー"), width=3),
                         dbc.Col(dcc.Dropdown(
                             id='llm-provider-selector',
                             options=[
                                 {'label': 'OpenAI', 'value': 'openai'},
                                 {'label': 'Azure OpenAI', 'value': 'azure'},
                                 {'label': 'Gemini', 'value': 'gemini'},
                                 {'label': 'Local LLM', 'value': 'local'},
                             ],
                             value='gemini' # 環境変数USE_AZUREに基づいて初期値を設定するコールバックが必要
                         ), width=9),
                     ], className="mb-3"),
                     # Local LLM 用のアドレス入力 (プロバイダー選択に応じて表示)
                     html.Div([
                         dbc.Row([
                             dbc.Col(dbc.Label("Local LLM Address"), width=3),
                             dbc.Col(dcc.Input(id='local-llm-address', type='text', placeholder='例: localhost:11434', className="form-control"), width=9)
                         ], className="mb-3")
                     ], id='local-llm-address-div', style={'display': 'none'}),

                     dbc.Row([
                         dbc.Col(dbc.Label("使用モデル"), width=3),
                         dbc.Col(dcc.Dropdown(id='llm-model-selector', options=[], placeholder="モデルを選択..."), width=9), # コールバックで更新
                     ], className="mb-3"),
                     dbc.Row([
                         dbc.Col(dbc.Label("並列実行数"), width=3),
                         dbc.Col(dcc.Input(id='llm-workers', type='number', value=3, min=1, step=1, className="form-control"), width=9),
                     ], className="mb-3"),
                      dbc.Row([
                         dbc.Col(dbc.Label("埋め込み処理"), width=3),
                         dbc.Col(dbc.Checklist(
                             options=[{'label': ' サーバー内部で実行 (APIコスト削減)', 'value': 'local_embed'}],
                             value=[], # デフォルトはAPIを使う
                             id='embedding-location-checkbox',
                             inline=True,
                             switch=True,
                         ), width=9),
                     ], className="mb-3", id='embedding-location-row'), # Local LLM選択時は非表示にする

                     # 各プロンプト編集エリア
                     html.Div([
                         html.H5("プロンプト編集", className="mt-4 mb-3"),
                         dbc.Label("意見抽出プロンプト"),
                         dcc.Textarea(id='prompt-extraction', value=extraction_prompt, className="form-control mb-2", rows=8),
                         dbc.Label("初期ラベリングプロンプト"),
                         dcc.Textarea(id='prompt-initial-labelling', value=initial_labelling_prompt, className="form-control mb-2", rows=8),
                         dbc.Label("統合ラベリングプロンプト"),
                         dcc.Textarea(id='prompt-merge-labelling', value=merge_labelling_prompt, className="form-control mb-2", rows=8),
                         dbc.Label("概要生成プロンプト"),
                         dcc.Textarea(id='prompt-overview', value=overview_prompt, className="form-control mb-2", rows=8),
                     ])
                 ])
            ])
        ], start_collapsed=True),


        # 実行ボタンとステータス表示
        dbc.Button("レポート作成を開始", id='create-report-button', color="primary", size="lg", className="mt-4", n_clicks=0),
        dbc.Spinner(html.Div(id='pipeline-status-output', className="mt-3")), # 実行状況表示用

    ], className="mt-4")

# コールバックはこのファイルではなく、callbacks/report_creator_callbacks.py に記述することを推奨