import multiprocessing
import time
import os
import json
from hierarchical_main import run_pipeline # hierarchical_main.pyに関数化されたパイプライン実行関数があると仮定
from hierarchical_utils import update_status # ステータス更新用ユーティリティ

def _run_pipeline_process(config):
    """別プロセスでパイプラインを実行する関数"""
    try:
        print(f"Pipeline process started for {config.get('output_dir', 'unknown')}")
        # hierarchical_main.py の実行関数を呼び出す
        run_pipeline(config)
        print(f"Pipeline process finished successfully for {config.get('output_dir', 'unknown')}")
    except Exception as e:
        print(f"Pipeline process failed for {config.get('output_dir', 'unknown')}: {e}")
        # エラー情報をstatus.jsonに書き込む (terminationで処理される想定だが念のため)
        error_info = {
            "status": "error",
            "end_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "error": f"{type(e).__name__}: {e}",
            "error_stack_trace": traceback.format_exc(),
        }
        try:
            # configオブジェクト自体を更新できない可能性があるため、ファイルパスから更新
            status_file = os.path.join('outputs', config['output_dir'], 'hierarchical_status.json')
            if os.path.exists(status_file):
                 with open(status_file, 'r+') as f:
                     status_data = json.load(f)
                     status_data.update(error_info)
                     f.seek(0)
                     json.dump(status_data, f, indent=2)
                     f.truncate()
            else: # まだ status.json がなければ作成
                 config.update(error_info) # 元のconfigを更新して保存試行 (不確実)
                 with open(status_file, "w") as f:
                     json.dump(config, f, indent=2)

        except Exception as update_e:
             print(f"Failed to update status file with error info: {update_e}")


def run_pipeline_async(config):
    """パイプラインを非同期で実行開始する"""
    # 実行前にステータスを更新 (initialization相当の処理をここで行う)
    output_dir = config["output_dir"]
    if not os.path.exists(f"outputs/{output_dir}"):
        os.makedirs(f"outputs/{output_dir}")

    # 既存のステータスファイル読み込み試行 (リトライなどのため)
    previous = None
    status_file_path = f"outputs/{output_dir}/hierarchical_status.json"
    if os.path.exists(status_file_path):
        try:
            with open(status_file_path) as f:
                previous = json.load(f)
            config["previous"] = previous # utils.decide_what_to_run で使えるように
        except Exception as e:
             print(f"Could not load previous status file: {e}")

    # 実行計画を作成 (hierarchical_utils.pyから呼び出し)
    from hierarchical_utils import decide_what_to_run
    plan = decide_what_to_run(config, previous) # この関数もファイルI/Oではなく辞書を扱えるように修正が必要かも

    # ステータスを 'running' に更新して保存
    initial_status = {
        "plan": plan,
        "status": "running",
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), # ISO 8601 format
        "completed_jobs": [],
        # config全体を含めると巨大になる可能性があるため、必要な情報だけ含めるか検討
        # "config": config # 必要ならconfig全体を含める
    }
    update_status(config, initial_status) # update_statusもファイルパスベースではなく辞書を扱えると良い

    # 別プロセスでパイプラインを実行
    process = multiprocessing.Process(target=_run_pipeline_process, args=(config,))
    process.start()
    print(f"Started background process for report: {config.get('output_dir', 'unknown')} (PID: {process.pid})")

    # 注意: この関数はすぐにリターンし、バックグラウンドで処理が続行されます。
    # Dash側では dcc.Interval などを使って定期的にステータスを確認する必要があります。