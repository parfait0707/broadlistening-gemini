import argparse
import sys
import json # 追加
import os   # 追加

from hierarchical_utils import initialization, run_step, termination, update_status
from steps.embedding import embedding
from steps.extraction import extraction
from steps.hierarchical_aggregation import hierarchical_aggregation
from steps.hierarchical_clustering import hierarchical_clustering
from steps.hierarchical_initial_labelling import hierarchical_initial_labelling
from steps.hierarchical_merge_labelling import hierarchical_merge_labelling
from steps.hierarchical_overview import hierarchical_overview
#from steps.hierarchical_visualization import hierarchical_visualization


def run_pipeline(config):
    """
    設定オブジェクトを受け取り、パイプラインを実行する関数。
    hierarchical_main.py の main() 関数のロジックを移植・改修。
    """
    # initialization で行っていたファイル読み込み以外の処理を実行
    # (例: デフォルト値の設定、ソースコード読み込みなど)
    # 注意: initialization関数自体を改修して、設定オブジェクトを直接扱えるようにする方が良い
    output_dir = config["output_dir"]
    if not os.path.exists(f"outputs/{output_dir}"):
        os.makedirs(f"outputs/{output_dir}")

    # ステータス更新 (pipeline_runner側で開始時に更新済みかもしれないが念のため)
    update_status(config, {"status": "running"}) # configが最新の状態を反映している前提

    try:
        run_step("extraction", extraction, config)
        run_step("embedding", embedding, config)
        run_step("hierarchical_clustering", hierarchical_clustering, config)
        run_step("hierarchical_initial_labelling", hierarchical_initial_labelling, config)
        run_step("hierarchical_merge_labelling", hierarchical_merge_labelling, config)
        run_step("hierarchical_overview", hierarchical_overview, config)
        run_step("hierarchical_aggregation", hierarchical_aggregation, config)
        # run_step("hierarchical_visualization", hierarchical_visualization, config) # Dashアプリ側で実行

        termination(config) # 正常終了時のステータス更新
    except Exception as e:
        # エラー発生時のステータス更新
        termination(config, error=e)
        # エラーを再発生させて呼び出し元に伝える
        raise e

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the annotation pipeline with optional flags.")
    parser.add_argument("config", help="Path to config JSON file that defines the pipeline execution.")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-run all steps regardless of previous execution.",
    )
    parser.add_argument(
        "-o",
        "--only",
        type=str,
        help="Run only the specified step (e.g., extraction, embedding, clustering, etc.).",
    )
    parser.add_argument(
        "--skip-interaction",
        action="store_true",
        help="Skip the interactive confirmation prompt and run pipeline immediately.",
    )

    # parser.add_argument(
    #     "--without-html",
    #     action="store_true",
    #     help="Skip the html output.",
    # )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Convert argparse namespace to sys.argv format for compatibility
    new_argv = [sys.argv[0], args.config]
    if args.force:
        new_argv.append("-f")
    if args.only:
        new_argv.extend(["-o", args.only])
    if args.skip_interaction:
        new_argv.append("-skip-interaction")
    # if args.without_html:
    #     new_argv.append("--without-html")

    config = initialization(new_argv)
    
    # 関数化されたパイプライン実行を呼び出す
    run_pipeline(config)



if __name__ == "__main__":
    main()