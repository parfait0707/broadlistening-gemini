import os
import json
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# 仮のレポートデータ取得関数 (実際には outputs ディレクトリをスキャン)
def get_reports():
    outputs_dir = 'outputs'
    reports = []
    if not os.path.exists(outputs_dir):
        return []
    for slug in os.listdir(outputs_dir):
        status_path = os.path.join(outputs_dir, slug, 'hierarchical_status.json')
        result_path = os.path.join(outputs_dir, slug, 'hierarchical_result.json')
        report_info = {"slug": slug, "title": slug, "status": "unknown", "description": "", "created_at": None} # 基本情報

        # result.jsonがあればそちらから情報を取得
        if os.path.exists(result_path):
             try:
                 with open(result_path, 'r', encoding='utf-8') as f:
                     data = json.load(f)
                     report_info["title"] = data.get("config", {}).get("question", slug)
                     report_info["description"] = data.get("config", {}).get("intro", "")
                     report_info["status"] = "ready" # resultがあれば完了とみなす
                     # 作成日時はstatus.jsonから取得推奨
             except Exception as e:
                 print(f"Error reading result.json for {slug}: {e}")

        # status.jsonがあればstatusと作成日時を上書き
        if os.path.exists(status_path):
            try:
                with open(status_path, 'r', encoding='utf-8') as f:
                    status_data = json.load(f)
                    # result.jsonがない場合のみstatusを更新
                    if report_info["status"] == "unknown":
                         report_info["status"] = status_data.get("status", "unknown")
                    report_info["created_at"] = status_data.get("start_time") # または end_time
            except Exception as e:
                print(f"Error reading status.json for {slug}: {e}")

        reports.append(report_info)

    # 作成日時でソート (新しい順)
    reports.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return reports

def layout():
    reports = get_reports()

    if not reports:
        return dbc.Container([
            html.H2("レポート一覧"),
            html.P("作成されたレポートはありません。"),
            dbc.Button("レポートを作成", href="/create", color="primary", className="mt-3")
        ], className="mt-4")

    report_cards = []
    for report in reports:
        card = dbc.Card(
            dbc.CardBody([
                html.H5(report['title'], className="card-title"),
                html.P(f"ID: {report['slug']}", className="text-muted small"),
                html.P(f"ステータス: {report['status']}", className="card-text"),
                html.P(f"作成日時: {report.get('created_at', 'N/A')}", className="card-text small"),
                 # ステータスに応じてリンク先やスタイルを変更
                dbc.Button("レポートを見る" if report['status'] == 'ready' else "詳細",
                           href=f"/report/{report['slug']}" if report['status'] == 'ready' else '#',
                           color="primary",
                           disabled=report['status'] != 'ready', # 完了以外は無効
                           size="sm",
                           className="mt-2")
            ]),
            className="mb-3"
        )
        report_cards.append(card)

    return dbc.Container([
        html.H2("レポート一覧", className="mb-4"),
        dbc.Button("新しいレポートを作成", href="/create", color="success", className="mb-4"),
        *report_cards
    ], className="mt-4")

# コールバックはこのファイルではなく、callbacks/report_list_callbacks.py に記述することを推奨