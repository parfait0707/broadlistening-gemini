import base64
import io
import pandas as pd
import json
import dash
from dash import html, dcc, callback, Input, Output, State, PreventUpdate
import dash_bootstrap_components as dbc

# パイプライン実行サービスをインポート
from services import pipeline_runner
from hierarchical_specs import specs # デフォルト値取得用

# 仮のCSV/Spreadsheet読み込み・カラム選択ロジック
def parse_input_data(contents, filename, input_type, url):
    if input_type == 'file' and contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # 文字コード自動判定 (chardet推奨)
            # import chardet
            # encoding = chardet.detect(decoded)['encoding'] or 'utf-8'
            encoding = 'utf-8' # または shift_jis など固定で試す
            df = pd.read_csv(io.StringIO(decoded.decode(encoding)))
            return df, df.columns.tolist(), None # DataFrame, カラムリスト, エラーなし
        except Exception as e:
            return None, [], f'ファイル読み込みエラー: {e}'
    elif input_type == 'spreadsheet' and url:
        try:
            # TODO: services.spreadsheet.py を呼び出してデータを取得
            # df = spreadsheet_service.fetch_data(url)
            # return df, df.columns.tolist(), None
            print(f"Spreadsheet fetch for URL: {url}") # 仮実装
            # 仮のデータフレームとカラム
            dummy_data = {'コメント': ['コメント1', 'コメント2'], 'ID': [1, 2]}
            df = pd.DataFrame(dummy_data)
            return df, df.columns.tolist(), None
        except Exception as e:
            return None, [], f'Spreadsheet読み込みエラー: {e}'
    return None, [], None

# 推奨クラスタ数計算 (仮)
def calculate_recommended_clusters(num_comments):
    if not num_comments or num_comments <= 0:
        return "データ読み込み後に推奨値が表示されます"
    # リポジトリBのロジックを参考に実装
    lv1 = max(2, min(10, round(num_comments**(1/3))))
    lv2 = max(2, min(1000, round(lv1 * lv1)))
    # 必要に応じて調整
    if lv2 < lv1 * 2:
        lv2 = lv1 * 2
    return f"推奨値: {lv1}, {lv2}, ..." # 文字列として返す例

def register_callbacks(app):

    # 入力タイプに応じて表示を切り替え
    @app.callback(
        [Output('file-upload-div', 'style'), Output('spreadsheet-div', 'style')],
        Input('input-type-selector', 'value')
    )
    def toggle_input_type(input_type):
        if input_type == 'file':
            return {'display': 'block'}, {'display': 'none'}
        elif input_type == 'spreadsheet':
            return {'display': 'none'}, {'display': 'block'}
        return {'display': 'none'}, {'display': 'none'}

    # ファイルアップロード or Spreadsheet取得 -> カラム選択表示
    @app.callback(
        [Output('output-upload-state', 'children'),
         Output('spreadsheet-load-state', 'children'),
         Output('column-selection-div', 'style'),
         Output('comment-column-selector', 'options'),
         Output('comment-column-selector', 'value'),
         Output('cluster-recommendation-output', 'children', allow_duplicate=True)], # allow_duplicate=True
        [Input('upload-csv', 'contents'),
         Input('fetch-spreadsheet-button', 'n_clicks')],
        [State('upload-csv', 'filename'),
         State('input-type-selector', 'value'),
         State('spreadsheet-url', 'value')],
         prevent_initial_call=True)
    def update_column_selector(contents, n_clicks, filename, input_type, url):
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

        df = None
        columns = []
        error_msg = None
        num_comments = 0

        if triggered_id == 'upload-csv' and contents:
            df, columns, error_msg = parse_input_data(contents, filename, 'file', None)
            if error_msg:
                return dbc.Alert(error_msg, color="danger"), None, {'display': 'none'}, [], None, calculate_recommended_clusters(0)
            upload_msg = dbc.Alert(f"ファイル '{filename}' を読み込みました。", color="success") if filename else None
            num_comments = len(df) if df is not None else 0
            return upload_msg, None, {'display': 'block'}, [{'label': col, 'value': col} for col in columns], columns[0] if columns else None, calculate_recommended_clusters(num_comments) # 最初のカラムをデフォルト選択

        elif triggered_id == 'fetch-spreadsheet-button' and n_clicks:
            df, columns, error_msg = parse_input_data(None, None, 'spreadsheet', url)
            if error_msg:
                return None, dbc.Alert(error_msg, color="danger"), {'display': 'none'}, [], None, calculate_recommended_clusters(0)
            load_msg = dbc.Alert(f"Spreadsheetから {len(df) if df is not None else 0} 件のデータを取得しました。", color="success")
            num_comments = len(df) if df is not None else 0
            return None, load_msg, {'display': 'block'}, [{'label': col, 'value': col} for col in columns], columns[0] if columns else None, calculate_recommended_clusters(num_comments)

        raise PreventUpdate

    # レポート作成ボタンクリック時の処理
    @app.callback(
        Output('pipeline-status-output', 'children'),
        Input('create-report-button', 'n_clicks'),
        [State('report-question', 'value'),
         State('report-intro', 'value'),
         State('report-slug', 'value'),
         State('input-type-selector', 'value'),
         State('upload-csv', 'contents'),
         State('upload-csv', 'filename'),
         State('spreadsheet-url', 'value'),
         State('comment-column-selector', 'value'),
         State('cluster-levels-input', 'value'),
         State('llm-provider-selector', 'value'),
         State('local-llm-address', 'value'),
         State('llm-model-selector', 'value'),
         State('llm-workers', 'value'),
         State('embedding-location-checkbox', 'value'),
         State('prompt-extraction', 'value'),
         State('prompt-initial-labelling', 'value'),
         State('prompt-merge-labelling', 'value'),
         State('prompt-overview', 'value')
         ],
        prevent_initial_call=True)
    def start_pipeline(n_clicks, question, intro, slug, input_type,
                       csv_contents, csv_filename, spreadsheet_url, selected_column,
                       cluster_levels_str, provider, local_llm_address, model, workers,
                       embedding_location, prompt_extraction, prompt_initial,
                       prompt_merge, prompt_overview):
        if n_clicks == 0:
            raise PreventUpdate

        # --- 入力値バリデーション ---
        if not all([question, intro, slug, selected_column, cluster_levels_str, provider, model, workers]):
             return dbc.Alert("必須項目が入力されていません。", color="danger")
        if not re.match(r'^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$', slug):
             return dbc.Alert("レポートIDは半角英数字とハイフンのみ使用可能です。", color="danger")
        if input_type == 'file' and not csv_contents:
            return dbc.Alert("CSVファイルがアップロードされていません。", color="danger")
        if input_type == 'spreadsheet' and not spreadsheet_url: # TODO: spreadsheetImported の状態も考慮
             return dbc.Alert("SpreadsheetのURLが入力されていないか、データが取得されていません。", color="danger")

        try:
            cluster_nums = [int(x.strip()) for x in cluster_levels_str.split(',') if x.strip().isdigit()]
            if not cluster_nums or len(cluster_nums) < 2 or not all(cluster_nums[i] < cluster_nums[i+1] for i in range(len(cluster_nums)-1)):
                raise ValueError("クラスタ数は2つ以上の昇順の整数をカンマ区切りで入力してください。")
        except ValueError as e:
            return dbc.Alert(f"クラスタ数の形式エラー: {e}", color="danger")

        # --- 設定オブジェクトの構築 ---
        # specs.jsonからデフォルト値を取得・マージ
        config = {}
        try:
            with open("./hierarchical_specs.json", 'r') as f:
                default_specs = json.load(f)
                for step_spec in default_specs:
                    step_name = step_spec['step']
                    config[step_name] = step_spec.get('options', {})
                    # LLM関連のデフォルトも設定 (必要なら)
                    if step_spec.get('use_llm') and 'model' not in config[step_name]:
                         config[step_name]['model'] = model # トップレベルのモデルをデフォルトに
                    if step_spec.get('use_llm') and 'prompt' not in config[step_name]:
                         # デフォルトプロンプトを読み込む処理 (hierarchical_utils.py参考)
                         try:
                            file = config[step_name].get("prompt_file", "default")
                            with open(f"prompts/{step_name}/{file}.txt", encoding='utf-8') as pf:
                                config[step_name]["prompt"] = pf.read()
                         except FileNotFoundError:
                            config[step_name]["prompt"] = "" # ファイルがない場合
        except Exception as e:
            print(f"Error loading default specs: {e}")
            # specs.jsonがなくても最低限動作するようにフォールバックが必要かも

        # ユーザー入力で上書き
        config['name'] = slug # レポートIDをnameに流用
        config['question'] = question
        config['intro'] = intro
        config['input'] = slug # 入力ファイル名もslugに合わせる (後でファイル処理が必要)
        config['provider'] = provider
        config['local_llm_address'] = local_llm_address if provider == 'local' else None
        config['model'] = model # 全体のデフォルトモデル
        config['is_embedded_at_local'] = 'local_embed' in embedding_location or provider == 'local'
        config['is_pubcom'] = True # Dashアプリでは常にTrueで良いか？

        config['extraction']['workers'] = workers
        config['extraction']['limit'] = 100000 # UIから制限しない場合は大きな値を設定
        config['extraction']['model'] = model # ステップごとにも設定
        config['extraction']['prompt'] = prompt_extraction

        config['hierarchical_clustering']['cluster_nums'] = cluster_nums

        config['hierarchical_initial_labelling']['workers'] = workers
        config['hierarchical_initial_labelling']['model'] = model
        config['hierarchical_initial_labelling']['prompt'] = prompt_initial

        config['hierarchical_merge_labelling']['workers'] = workers
        config['hierarchical_merge_labelling']['model'] = model
        config['hierarchical_merge_labelling']['prompt'] = prompt_merge

        config['hierarchical_overview']['model'] = model
        config['hierarchical_overview']['prompt'] = prompt_overview

        # --- 入力データの準備 ---
        input_file_path = os.path.join('inputs', f"{slug}.csv")
        try:
            if input_type == 'file':
                content_type, content_string = csv_contents.split(',')
                decoded = base64.b64decode(content_string)
                # TODO: 読み込んだDFから選択されたカラムのみを'comment-body'としてCSV保存
                # df = pd.read_csv(io.StringIO(decoded.decode('utf-8'))) # 文字コード判定推奨
                # df_out = df[[selected_column]].rename(columns={selected_column: 'comment-body'})
                # df_out['comment-id'] = df_out.index # 仮のID
                # df_out.to_csv(input_file_path, index=False)
                with open(input_file_path, 'wb') as f: # 仮: バイナリ書き込み
                     f.write(decoded)
                print(f"Saved uploaded CSV to {input_file_path}")

            elif input_type == 'spreadsheet':
                 # TODO: spreadsheet_service から取得したデータをCSV保存
                 # df = spreadsheet_service.fetch_data(spreadsheet_url) # 再取得またはStoreから取得
                 # df_out = df[[selected_column]].rename(columns={selected_column: 'comment-body'})
                 # df_out['comment-id'] = df.get('id', df.index) # IDがあれば使う
                 # df_out.to_csv(input_file_path, index=False)
                 print(f"Saving spreadsheet data for {slug} to {input_file_path}") # 仮実装
                 with open(input_file_path, 'w') as f:
                     f.write("comment-id,comment-body\n1,dummy comment from sheet")

        except Exception as e:
            return dbc.Alert(f"入力データの準備中にエラーが発生しました: {e}", color="danger")

        # --- パイプライン実行 ---
        try:
            print(f"Starting pipeline for report: {slug}")
            pipeline_runner.run_pipeline_async(config)
            # 最初のステータスメッセージ
            initial_status_msg = f"レポート '{slug}' の作成を開始しました。完了まで数分～数十分かかることがあります。"
            # TODO: dcc.Intervalを追加して定期的にステータスを確認・表示する
            return dbc.Alert(initial_status_msg, color="info")
        except Exception as e:
            return dbc.Alert(f"パイプラインの開始に失敗しました: {e}", color="danger")

    # LLMプロバイダーに応じてモデル選択肢を更新
    @app.callback(
        [Output('llm-model-selector', 'options'),
         Output('llm-model-selector', 'value'),
         Output('local-llm-address-div', 'style'),
         Output('embedding-location-row', 'style'),
         Output('embedding-location-checkbox', 'value')],
        [Input('llm-provider-selector', 'value'),
         Input('local-llm-address', 'value')], # アドレス変更でもトリガー (ボタンは別途)
        [State('llm-model-selector', 'value')] # 現在選択中のモデル
    )
    def update_model_options(selected_provider, address, current_model):
        options = []
        default_model = None
        local_llm_style = {'display': 'none'}
        embedding_row_style = {'display': 'flex'} # デフォルト表示
        embedding_local_value = [] # デフォルトはAPI

        if selected_provider == 'openai':
            options = [{'label': 'GPT-4o mini', 'value': 'gpt-4o-mini'},
                       {'label': 'GPT-4o', 'value': 'gpt-4o'}]
            default_model = 'gpt-4o-mini'
        elif selected_provider == 'azure':
             # Azureの場合はデプロイ名を使う想定だが、選択肢としてはモデル名を見せる
             # 実際のAPIコールではconfig['model']ではなくdeployment_nameを使うようにllm.py側で調整
             options = [{'label': 'GPT-4o mini (デプロイ名要確認)', 'value': 'gpt-4o-mini'},
                        {'label': 'GPT-4o (デプロイ名要確認)', 'value': 'gpt-4o'}]
             default_model = 'gpt-4o-mini'
        elif selected_provider == 'gemini':
             # TODO: Geminiの利用可能なモデルリストをAPI経由で取得する？
             options = [{'label': 'Gemini 1.5 Flash', 'value': 'gemini-1.5-flash-latest'},
                        {'label': 'Gemini 1.5 Pro', 'value': 'gemini-1.5-pro-latest'}]
             default_model = 'gemini-1.5-flash-latest'
        elif selected_provider == 'local':
             # TODO: APIを叩いて取得する or 固定リスト
             # ここでは仮のリスト
             options = [{'label': 'Llama3 (Local)', 'value': 'llama3'}, {'label': 'Mistral (Local)', 'value': 'mistral'}]
             default_model = options[0]['value'] if options else None
             local_llm_style = {'display': 'block'}
             embedding_row_style = {'display': 'none'} # Local LLMなら埋め込みもLocal想定
             embedding_local_value = ['local_embed']

        # 現在選択中のモデルが新しいオプションリストにあれば維持、なければデフォルト
        final_model_value = default_model
        if current_model and any(opt['value'] == current_model for opt in options):
             final_model_value = current_model

        return options, final_model_value, local_llm_style, embedding_row_style, embedding_local_value

    # TODO: レポートIDのバリデーション用コールバック
    # TODO: 推奨クラスタ数表示用コールバック
    # TODO: パイプラインステータス表示用コールバック (dcc.Interval)