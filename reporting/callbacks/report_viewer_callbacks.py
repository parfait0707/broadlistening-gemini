import json
from dash import callback, Output, Input, State, Patch, PreventUpdate, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# グラフ描画コンポーネントをインポート
from reporting.components import charts
# ページレイアウトからロード関数をインポート (循環参照を避けるため)
from reporting.pages.report_viewer import load_report_data

def register_callbacks(app):

    # チャートタイプに応じて設定表示を切り替え、グラフを更新
    @app.callback(
        [Output('graph-output', 'children'),
         Output('dense-settings-collapse', 'is_open'),
         Output('treemap-controls-collapse', 'is_open'),
         Output('treemap-back-button', 'disabled'),
         Output('chart-config-store', 'data', allow_duplicate=True)], # Storeも更新
        [Input('chart-type-selector', 'value'),
         Input('density-slider', 'value'),
         Input('min-value-slider', 'value'),
         Input('show-labels-checkbox', 'value'),
         Input('treemap-graph', 'clickData'), # Treemapクリックを検知
         Input('treemap-back-button', 'n_clicks')],
        [State('report-data-store', 'data'),
         State('chart-config-store', 'data')], # 現在の設定を取得
        prevent_initial_call=True
    )
    def update_graph_and_settings(
        chart_type, density_pct, min_value, show_labels, click_data, back_clicks,
        report_data, current_chart_config):

        if not report_data:
            raise PreventUpdate

        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

        # パッチオブジェクトで一部だけ更新
        patched_config = Patch()
        new_treemap_level = current_chart_config.get('treemap_level', '0')
        new_chart_type = current_chart_config.get('selected_chart', 'scatterAll')

        # チャートタイプが変更された場合
        if triggered_id == 'chart-type-selector':
            new_chart_type = chart_type
            patched_config['selected_chart'] = new_chart_type
            # Treemapに切り替えたらレベルをリセット
            if new_chart_type == 'treemap':
                new_treemap_level = '0'
                patched_config['treemap_level'] = new_treemap_level

        # 密度フィルター関連の設定が変更された場合
        if triggered_id in ['density-slider', 'min-value-slider']:
             patched_config['max_density'] = density_pct / 100.0 # パーセントから小数へ
             patched_config['min_value'] = min_value

        # ラベル表示が変更された場合
        if triggered_id == 'show-labels-checkbox':
            patched_config['show_labels'] = show_labels

        # Treemapがクリックされた場合
        if triggered_id == 'treemap-graph' and click_data and new_chart_type == 'treemap':
             clicked_id = click_data['points'][0].get('id') # IDを取得
             if clicked_id and clicked_id != new_treemap_level:
                 new_treemap_level = clicked_id
                 patched_config['treemap_level'] = new_treemap_level

        # Treemapの戻るボタンが押された場合
        if triggered_id == 'treemap-back-button' and new_chart_type == 'treemap':
             current_parent = None
             for cluster in report_data.get('clusters', []):
                 if cluster.get('id') == new_treemap_level:
                     current_parent = cluster.get('parent')
                     break
             if current_parent is not None and current_parent != "": # ルート("")でなければ親に戻る
                 new_treemap_level = current_parent
                 patched_config['treemap_level'] = new_treemap_level


        # 設定の更新を適用 (Stateから取得した現在の値を使う)
        final_chart_config = {**current_chart_config, **patched_config}

        # 設定に基づいてUI表示を決定
        dense_settings_open = final_chart_config['selected_chart'] == 'scatterDensity'
        treemap_controls_open = final_chart_config['selected_chart'] == 'treemap'
        treemap_back_disabled = final_chart_config['treemap_level'] == '0' # ルートなら無効

        # グラフを生成
        graph_component = None
        if final_chart_config['selected_chart'] == 'scatterAll':
            graph_component = charts.create_scatter_chart(
                report_data,
                target_level=1, # Level 1固定
                show_labels=final_chart_config['show_labels']
            )
        elif final_chart_config['selected_chart'] == 'scatterDensity':
             # 密度フィルターを適用したデータを生成（これはコールバック内で行うか、専用関数を呼ぶ）
             # TODO: getDenseClusters相当のロジックをPythonで実装し、filtered_clusters を取得
             filtered_clusters = get_filtered_clusters(
                 report_data['clusters'],
                 final_chart_config['max_density'],
                 final_chart_config['min_value']
             )
             is_dense_enabled = len(filtered_clusters) > 0 # フィルター結果があるか
             patched_config['is_dense_group_enabled'] = is_dense_enabled # Storeを更新
             final_chart_config['is_dense_group_enabled'] = is_dense_enabled

             if is_dense_enabled:
                 graph_component = charts.create_scatter_chart(
                     {'clusters': filtered_clusters, 'arguments': report_data['arguments']}, # フィルター結果で描画
                     target_level=max(c['level'] for c in filtered_clusters), # 最深レベル
                     show_labels=final_chart_config['show_labels']
                 )
             else:
                 graph_component = html.Div("指定された条件で表示できる濃い意見グループはありません。")

        elif final_chart_config['selected_chart'] == 'treemap':
            graph_component = charts.create_treemap_chart(
                report_data,
                current_level_id=final_chart_config['treemap_level']
            )

        return graph_component, dense_settings_open, treemap_controls_open, treemap_back_disabled, final_chart_config

# 密度フィルター処理（仮実装）
def get_filtered_clusters(clusters, max_density_pct, min_value):
    if not clusters:
        return []
    deepest_level = max(c.get('level', 0) for c in clusters if 'level' in c)
    if deepest_level == 0:
         return [] # レベル0しかない場合は空

    deepest_clusters = [c for c in clusters if c.get('level') == deepest_level]

    # density_rank_percentileでフィルタリング
    filtered = [
        c for c in deepest_clusters
        if c.get('density_rank_percentile', 1.1) <= max_density_pct # 1.1はデフォルトで除外される値
        and c.get('value', 0) >= min_value
    ]
    print(f"Filtering: level={deepest_level}, max_density={max_density_pct}, min_value={min_value}, count={len(filtered)}")
    return filtered
