import json
import logging
import os
import threading
import time  # 必要に応じて追加
from typing import Any, Dict, List, Type, Union # 型ヒントを修正・追加

import openai
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI # OpenAIErrorを追加 (APIConnectionErrorなどを含む)
from pydantic import BaseModel, Field

# tenacityをインポート
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log # ログ出力用のヘルパーを追加
)

DOTENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.env"))
load_dotenv(DOTENV_PATH)

# ログ設定
logging.basicConfig(level=logging.INFO) # INFOレベルに変更してリトライなども記録
logger = logging.getLogger(__name__)

# check env
use_azure = os.getenv("USE_AZURE", "false").lower() == "true" # booleanに変換
openai_api_key = os.getenv("OPENAI_API_KEY")

if use_azure:
    if not os.getenv("AZURE_CHATCOMPLETION_ENDPOINT"):
        raise RuntimeError("AZURE_CHATCOMPLETION_ENDPOINT environment variable is not set when USE_AZURE is true")
    if not os.getenv("AZURE_CHATCOMPLETION_DEPLOYMENT_NAME"):
        raise RuntimeError("AZURE_CHATCOMPLETION_DEPLOYMENT_NAME environment variable is not set when USE_AZURE is true")
    if not os.getenv("AZURE_CHATCOMPLETION_API_KEY"):
        raise RuntimeError("AZURE_CHATCOMPLETION_API_KEY environment variable is not set when USE_AZURE is true")
    if not os.getenv("AZURE_CHATCOMPLETION_VERSION"):
        raise RuntimeError("AZURE_CHATCOMPLETION_VERSION environment variable is not set when USE_AZURE is true")
    if not os.getenv("AZURE_EMBEDDING_ENDPOINT"):
        raise RuntimeError("AZURE_EMBEDDING_ENDPOINT environment variable is not set when USE_AZURE is true")
    if not os.getenv("AZURE_EMBEDDING_API_KEY"):
        raise RuntimeError("AZURE_EMBEDDING_API_KEY environment variable is not set when USE_AZURE is true")
    if not os.getenv("AZURE_EMBEDDING_VERSION"):
        raise RuntimeError("AZURE_EMBEDDING_VERSION environment variable is not set when USE_AZURE is true")
    if not os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"):
        raise RuntimeError("AZURE_EMBEDDING_DEPLOYMENT_NAME environment variable is not set when USE_AZURE is true")
elif not openai_api_key and "OPENAI_API_KEY" not in os.environ:
    # Azureを使わない場合、OpenAI APIキーまたはLocal LLM設定が必要
    # Local LLMを使う場合は provider='local' が指定されるはずなので、ここではOpenAIキーのみチェック
    # logger.warning("Neither USE_AZURE is true nor OPENAI_API_KEY is set. Assuming Local LLM might be used or key is set elsewhere.")
    # provider='local' が指定されない限りエラーにする方が安全かもしれない
    pass # ここでの厳密なチェックは request_to_chat_llm / request_to_embed に任せる


# --- リトライ設定 ---
# リトライ対象とする例外を指定 (OpenAI/Azure/汎用エラー)
RETRYABLE_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    openai.UnprocessableEntityError, # ローカルLLM等で発生しうる
    ConnectionError, # ローカルLLM接続エラー
)

# リトライの基本設定
RETRY_CONFIG = {
    "retry": retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    "wait": wait_exponential(multiplier=1, min=3, max=20), # 待機時間: 3s, 6s, 12s,... 最大20s
    "stop": stop_after_attempt(4), # 最大4回試行 (初回の試行含む)
    "reraise": True, # リトライ失敗時に元の例外を再発生させる
    "before_sleep": before_sleep_log(logger, logging.WARNING) # tenacityのログヘルパーを使用
}
# --- ここまでリトライ設定 ---

@retry(**RETRY_CONFIG)
def request_to_openai_chatcompletion(
    messages: list[dict],
    model: str = "gpt-4o",
    is_json: bool = False,
    json_schema: dict | type[BaseModel] | None = None,
) -> Union[str, Dict[str, Any]]: # 戻り値の型ヒント修正
    """OpenAI API (非Azure) にリクエストを送信する関数"""
    if not openai_api_key:
         raise ValueError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=openai_api_key)
    try:
        response_format = None
        tools = None
        tool_choice = None

        if isinstance(json_schema, type) and issubclass(json_schema, BaseModel):
            # Pydanticモデルを使う場合 (tools APIを使用)
            tools = [{"type": "function", "function": {"name": json_schema.__name__, "parameters": json_schema.model_json_schema()}}]
            tool_choice = {"type": "function", "function": {"name": json_schema.__name__}}
            logger.debug(f"Using tools API for Pydantic schema: {json_schema.__name__}")
        elif is_json or (json_schema and isinstance(json_schema, dict)):
            # 標準のJSONモードまたは辞書形式のJSONスキーマ
            response_format = {"type": "json_object"}
            # json_schemaがdictの場合、現状OpenAI APIでは直接スキーマ指定できないため、JSONモードにする
            if json_schema and isinstance(json_schema, dict):
                 logger.warning("Dictionary json_schema provided for OpenAI, using standard json_object mode.")


        payload: Dict[str, Any] = { # 型ヒント追加
            "model": model,
            "messages": messages,
            "temperature": 0,
            "n": 1,
            "seed": 0,
            "timeout": 30,
        }
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
        if tool_choice:
             payload["tool_choice"] = tool_choice

        response = client.chat.completions.create(**payload)

        # ツールコールがある場合の処理
        if response.choices[0].message.tool_calls:
             tool_calls = response.choices[0].message.tool_calls
             if tool_calls and tool_calls[0].function.arguments:
                 try:
                     # 厳密には json.loads が必要
                     return json.loads(tool_calls[0].function.arguments)
                 except json.JSONDecodeError as e:
                     logging.error(f"Failed to parse tool call arguments as JSON: {e}")
                     raise ValueError("Failed to parse structured response from OpenAI API (tool call)") from e
             else:
                  logging.error("OpenAI chat completion response contained tool_calls but no arguments.")
                  raise ValueError("Failed to parse structured response from OpenAI API (tool call)")
        # 通常のテキストレスポンス
        elif response.choices[0].message.content is not None:
             return response.choices[0].message.content
        else:
            logging.error("OpenAI chat completion response has neither content nor tool_calls.")
            raise ValueError("Invalid response from OpenAI API")

    # tenacityで捕捉される例外はここで再発生させる
    except RETRYABLE_EXCEPTIONS as e:
        raise e
    except openai.AuthenticationError as e:
        logging.error(f"OpenAI API authentication error: {str(e)}")
        raise
    except openai.BadRequestError as e:
        logging.error(f"OpenAI API bad request error: {str(e)}")
        raise
    except Exception as e: # 予期せぬエラー
        logging.error(f"An unexpected error occurred during OpenAI chat completion: {e}")
        raise


@retry(**RETRY_CONFIG)
def request_to_azure_chatcompletion(
    messages: list[dict],
    model: str, # model引数はAzureでは通常使わないが、互換性のために残す（内部でdeployment名を使用）
    is_json: bool = False,
    json_schema: dict | type[BaseModel] | None = None,
) -> Union[str, Dict[str, Any]]: # 戻り値の型ヒント修正
    azure_endpoint = os.getenv("AZURE_CHATCOMPLETION_ENDPOINT")
    deployment = os.getenv("AZURE_CHATCOMPLETION_DEPLOYMENT_NAME") # デプロイ名はここで取得
    api_key = os.getenv("AZURE_CHATCOMPLETION_API_KEY")
    api_version = os.getenv("AZURE_CHATCOMPLETION_VERSION")

    if not all([azure_endpoint, deployment, api_key, api_version]):
         raise ValueError("Azure environment variables are not fully set.")

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=azure_endpoint,
        api_key=api_key,
    )

    try:
        # Azure OpenAI の Pydantic スキーマ対応 (tools API を使用)
        if isinstance(json_schema, type) and issubclass(json_schema, BaseModel):
            tools = [{"type": "function", "function": {"name": json_schema.__name__, "parameters": json_schema.model_json_schema()}}]
            tool_choice = {"type": "function", "function": {"name": json_schema.__name__}}
            logger.debug(f"Using Azure tools API for Pydantic schema: {json_schema.__name__}")

            response = client.chat.completions.create(
                model=deployment, # Azureではデプロイ名を指定
                messages=messages,
                temperature=0,
                n=1,
                seed=0,
                tools=tools,
                tool_choice=tool_choice,
                timeout=30,
            )
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and tool_calls[0].function.arguments:
                 try:
                     return json.loads(tool_calls[0].function.arguments)
                 except json.JSONDecodeError as e:
                     logging.error(f"Failed to parse Azure tool call arguments as JSON: {e}")
                     raise ValueError("Failed to parse structured response from Azure API (tool call)") from e
            else:
                 logging.error("Azure chat completion response did not contain expected tool call arguments.")
                 raise ValueError("Failed to parse structured response from Azure API (tool call)")

        # Azure OpenAI の JSON Mode 対応
        else:
            response_format = None
            if is_json:
                # Azure OpenAI Service API version '2023-07-01-preview' 以降が必要
                response_format = {"type": "json_object"}
            # Azure OpenAIは現状 `json_schema` dict形式の response_format に非対応
            if json_schema and isinstance(json_schema, dict):
                 logger.warning("Dictionary json_schema provided for Azure OpenAI, using standard json_object mode if is_json=True.")
                 if is_json:
                      response_format = {"type": "json_object"} # is_jsonがTrueならJSONモードにする

            payload: Dict[str, Any] = { # 型ヒント追加
                "model": deployment, # Azureではデプロイ名を指定
                "messages": messages,
                "temperature": 0,
                "n": 1,
                "seed": 0,
                "timeout": 30,
            }
            if response_format:
                payload["response_format"] = response_format

            response = client.chat.completions.create(**payload)

            if response.choices[0].message.content is not None:
                 return response.choices[0].message.content
            else:
                 logging.error("Azure chat completion response has no content.")
                 raise ValueError("Invalid response from Azure API")

    # tenacityで捕捉される例外はここで再発生させる
    except RETRYABLE_EXCEPTIONS as e:
        raise e
    except openai.AuthenticationError as e:
        logging.error(f"Azure OpenAI API authentication error: {str(e)}")
        raise
    except openai.BadRequestError as e:
        logging.error(f"Azure OpenAI API bad request error: {str(e)}")
        raise
    except Exception as e: # 予期せぬエラー
        logging.error(f"An unexpected error occurred during Azure chat completion: {e}")
        raise

@retry(**RETRY_CONFIG)
def request_to_local_llm(
    messages: list[dict],
    model: str,
    is_json: bool = False,
    json_schema: dict | type[BaseModel] | None = None,
    address: str = "localhost:11434",
) -> Union[str, Dict[str, Any]]: # 戻り値の型ヒント修正
    """ローカルLLM（OllamaやLM StudioなどOpenAI互換API）にリクエストを送信する関数"""
    try:
        if ":" in address:
            host, port_str = address.split(":")
            port = int(port_str)
        else:
            host = address
            port = 11434  # デフォルトポート
    except ValueError:
        logger.warning(f"Invalid address format: {address}, using default localhost:11434")
        host = "localhost"
        port = 11434

    base_url = f"http://{host}:{port}/v1"
    logger.info(f"Attempting to connect to Local LLM at: {base_url}")

    try:
        client = OpenAI(
            base_url=base_url,
            api_key="not-needed",  # OllamaとLM Studio等は認証不要
        )

        response_format = None
        tools = None
        tool_choice = None

        if isinstance(json_schema, type) and issubclass(json_schema, BaseModel):
            # Local LLM が OpenAI の tools API をサポートしている場合
            try:
                tools = [{"type": "function", "function": {"name": json_schema.__name__, "parameters": json_schema.model_json_schema()}}]
                tool_choice = {"type": "function", "function": {"name": json_schema.__name__}}
                logger.debug(f"Attempting Local LLM tools API for Pydantic schema: {json_schema.__name__}")
            except Exception as schema_err:
                logger.warning(f"Failed to prepare Pydantic schema for Local LLM tools API, falling back to JSON mode if is_json=True: {schema_err}")
                if is_json:
                     response_format = {"type": "json_object"}
                tools = None
                tool_choice = None
        elif is_json or (json_schema and isinstance(json_schema, dict)):
             # Local LLMがJSONモードをサポートしている場合
             response_format = {"type": "json_object"}
             if json_schema and isinstance(json_schema, dict):
                 logger.warning("Dictionary json_schema provided for Local LLM, using standard json_object mode if supported.")

        payload: Dict[str, Any] = { # 型ヒント追加
            "model": model,
            "messages": messages,
            "temperature": 0,
            "n": 1,
            "seed": 0, # サポートされていない場合がある
            "timeout": 60, # ローカルは少し長めに
        }

        # サポート状況に応じてパラメータを追加
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
        if tool_choice:
             payload["tool_choice"] = tool_choice

        logger.debug(f"Sending payload to Local LLM: {payload}")
        response = client.chat.completions.create(**payload)
        logger.debug(f"Received response from Local LLM: {response}")

        # ツールコールがある場合の処理
        if response.choices[0].message.tool_calls:
             tool_calls = response.choices[0].message.tool_calls
             if tool_calls and tool_calls[0].function.arguments:
                 try:
                     # 厳密には json.loads が必要
                     return json.loads(tool_calls[0].function.arguments)
                 except json.JSONDecodeError as e:
                     logging.error(f"Failed to parse Local LLM tool call arguments as JSON: {e}")
                     raise ValueError("Failed to parse structured response from Local LLM API (tool call)") from e
             else:
                  logging.error("Local LLM chat completion response contained tool_calls but no arguments.")
                  raise ValueError("Failed to parse structured response from Local LLM API (tool call)")
        # 通常のテキストレスポンス
        elif response.choices[0].message.content is not None:
             return response.choices[0].message.content
        else:
            logging.error("Local LLM chat completion response has neither content nor tool_calls.")
            raise ValueError("Invalid response from Local LLM API")

    # tenacityで捕捉される例外はここで再発生させる
    except RETRYABLE_EXCEPTIONS as e:
        raise e
    except openai.APIConnectionError as e: # 接続エラーを捕捉
         logging.error(f"Could not connect to Local LLM at {base_url}: {e}")
         raise ConnectionError(f"Could not connect to Local LLM at {base_url}") from e
    except Exception as e: # 予期せぬエラー
        logging.error(f"An unexpected error occurred during Local LLM chat completion: {e}")
        raise


# --- request_to_chat_llm: プロバイダーに応じて適切な関数を呼び出す (Gemini削除) ---
def request_to_chat_llm(
    messages: list[dict],
    model: str = "gpt-4o", # デフォルトをgpt-4oに変更
    is_json: bool = False,
    json_schema: dict | type[BaseModel] | None = None,
    provider: str | None = None, # provider引数を追加
    local_llm_address: str | None = None, # local_llm_address引数を追加
) -> Union[str, Dict[str, Any]]: # 戻り値の型ヒント修正
    if provider is None:
        # providerが指定されていない場合、環境変数から判断
        provider = "azure" if use_azure else "openai"
        # OpenAIキーがなくAzureでもない場合はエラーにした方が安全か？
        if provider == "openai" and not openai_api_key:
            raise ValueError("Neither USE_AZURE is true nor OPENAI_API_KEY is set. Cannot determine default provider.")

    logger.info(f"Using LLM provider: {provider}, Model: {model}")

    if provider == "azure":
        return request_to_azure_chatcompletion(messages, model=model, is_json=is_json, json_schema=json_schema)
    elif provider == "openai":
        return request_to_openai_chatcompletion(messages, model, is_json, json_schema)
    elif provider == "local":
        if not local_llm_address:
             # 環境変数からフォールバックするなどの考慮も可能
             local_llm_address = os.getenv("LOCAL_LLM_ADDRESS", "localhost:11434")
             logger.warning(f"local_llm_address not provided, using default or environment variable: {local_llm_address}")
             # raise ValueError("local_llm_address is required for local provider")
        return request_to_local_llm(messages, model, is_json, json_schema, local_llm_address)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers are 'azure', 'openai', 'local'.")

# --- Embedding関連 ---
EMBEDDING_MODELS = [ # Geminiモデル削除
    "text-embedding-3-large",
    "text-embedding-3-small",
    # 必要であればローカル用モデル名を追加 (例: "local/your-embedding-model")
]

def _validate_embedding_model(model):
    # ローカルモデルの場合は検証をスキップするか、別のリストで管理する
    if model.startswith("local/"):
         return
    if model not in EMBEDDING_MODELS:
        logger.warning(f"Embedding model '{model}' not in known list: {EMBEDDING_MODELS}. Proceeding anyway.")
        # raise RuntimeError(f"Invalid embedding model: {model}, available models: {EMBEDDING_MODELS}")

@retry(**RETRY_CONFIG)
def request_to_openai_embed(args, model): # 関数名変更
    _validate_embedding_model(model)
    if not openai_api_key:
         raise ValueError("OPENAI_API_KEY environment variable is not set.")
    client = OpenAI(api_key=openai_api_key)
    try:
        response = client.embeddings.create(input=args, model=model)
        return [item.embedding for item in response.data]
    # tenacityで捕捉される例外はここで再発生させる
    except RETRYABLE_EXCEPTIONS as e:
        raise e
    except openai.AuthenticationError as e:
        logging.error(f"OpenAI API authentication error: {str(e)}")
        raise
    except openai.BadRequestError as e:
        logging.error(f"OpenAI API bad request error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during OpenAI embedding: {e}")
        raise

@retry(**RETRY_CONFIG)
def request_to_azure_embed(args, model):
    azure_endpoint = os.getenv("AZURE_EMBEDDING_ENDPOINT")
    api_key = os.getenv("AZURE_EMBEDDING_API_KEY")
    api_version = os.getenv("AZURE_EMBEDDING_VERSION")
    deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME") # Embeddings用のデプロイ名

    if not all([azure_endpoint, deployment, api_key, api_version]):
         raise ValueError("Azure Embedding environment variables are not fully set.")

    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=azure_endpoint,
        api_key=api_key,
    )
    try:
        response = client.embeddings.create(input=args, model=deployment) # Azureではデプロイ名を指定
        return [item.embedding for item in response.data]
    # tenacityで捕捉される例外はここで再発生させる
    except RETRYABLE_EXCEPTIONS as e:
        raise e
    except openai.AuthenticationError as e:
        logging.error(f"Azure OpenAI API authentication error: {str(e)}")
        raise
    except openai.BadRequestError as e:
        logging.error(f"Azure OpenAI API bad request error: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during Azure embedding: {e}")
        raise

@retry(**RETRY_CONFIG)
def request_to_local_llm_embed(args, model, address="localhost:11434"):
    """ローカルLLM（OllamaやLM StudioなどOpenAI互換API）を使用して埋め込みを取得する関数"""
    try:
        if ":" in address:
            host, port_str = address.split(":")
            port = int(port_str)
        else:
            host = address
            port = 11434
    except ValueError:
        logger.warning(f"Invalid address format: {address}, using default localhost:11434")
        host = "localhost"
        port = 11434

    base_url = f"http://{host}:{port}/v1"
    logger.info(f"Attempting to connect to Local LLM for embedding at: {base_url}")

    try:
        client = OpenAI(
            base_url=base_url,
            api_key="not-needed",
        )
        response = client.embeddings.create(input=args, model=model)
        embeds = [item.embedding for item in response.data]
        return embeds
    # tenacityで捕捉される例外はここで再発生させる
    except RETRYABLE_EXCEPTIONS as e:
        raise e
    except openai.APIConnectionError as e: # 接続エラーを捕捉
         logging.error(f"Could not connect to Local LLM at {base_url}: {e}")
         # ローカルLLM接続失敗時はローカルのsentence-transformerにフォールバック
         logging.warning("Falling back to local sentence-transformer embedding.")
         return request_to_local_embed(args)
    except Exception as e:
        logging.error(f"An unexpected error occurred during Local LLM embedding: {e}")
        # その他の予期せぬエラーでもフォールバック
        logging.warning("Falling back to local sentence-transformer embedding due to unexpected error.")
        return request_to_local_embed(args)


# --- Local Embedding (sentence-transformers) ---
__local_emb_model = None
__local_emb_model_loading_lock = threading.Lock()

def request_to_local_embed(args):
    """Sentence Transformersを使用してローカルで埋め込みを計算する"""
    global __local_emb_model
    with __local_emb_model_loading_lock:
        if __local_emb_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                 logger.error("sentence-transformers is not installed. Please install it to use local embeddings.")
                 raise
            logger.info("Loading local embedding model (sentence-transformers)...")
            # 推奨モデル例:
            # - paraphrase-multilingual-mpnet-base-v2 (多言語、軽量)
            # - intfloat/multilingual-e5-large (多言語、高性能)
            # - pkshatech/GLuCoSE-base-ja (日本語特化、高性能)
            model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" # リポジトリAのものをデフォルトに
            try:
                __local_emb_model = SentenceTransformer(model_name)
                logger.info(f"Local embedding model '{model_name}' loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load local embedding model '{model_name}': {e}")
                raise RuntimeError(f"Failed to load local embedding model: {e}") from e

    if __local_emb_model is None:
         raise RuntimeError("Local embedding model could not be loaded.")

    try:
        result = __local_emb_model.encode(args)
        return result.tolist()
    except Exception as e:
        logging.error(f"Error during local sentence-transformer embedding encoding: {e}")
        raise

# --- request_to_embed: プロバイダーに応じて適切な関数を呼び出す (Gemini削除) ---
def request_to_embed(args, model, is_embedded_at_local=False, provider=None, local_llm_address=None):
    if is_embedded_at_local:
        logger.info("Using local sentence-transformer embedding.")
        return request_to_local_embed(args)

    if provider is None:
        provider = "azure" if use_azure else "openai"
        if provider == "openai" and not openai_api_key:
             logger.warning("OPENAI_API_KEY not set, trying Azure embedding as fallback.")
             provider = "azure" # Azureもなければエラーになる

    logger.info(f"Using embedding provider: {provider}, Model: {model}")

    if provider == "azure":
        return request_to_azure_embed(args, model)
    elif provider == "openai":
        return request_to_openai_embed(args, model)
    elif provider == "local":
        if not local_llm_address:
             local_llm_address = os.getenv("LOCAL_LLM_ADDRESS", "localhost:11434")
             logger.warning(f"local_llm_address not provided for embedding, using default or environment variable: {local_llm_address}")
        # model引数はローカルサーバーで認識されるモデル名を渡す
        return request_to_local_llm_embed(args, model, local_llm_address)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}. Supported providers are 'azure', 'openai', 'local'.")


# --- テスト関数 (Gemini関連削除) ---
def _test_openai_chat():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    response = request_to_chat_llm(messages=messages, provider="openai", model="gpt-4o-mini")
    print("OpenAI Chat Response:", response)

def _test_azure_chat():
    if not use_azure:
        print("Skipping Azure chat test as USE_AZURE is not true.")
        return
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Japan?"},
    ]
    # Azureの場合、model引数は無視され、環境変数のデプロイ名が使われる
    response = request_to_chat_llm(messages=messages, provider="azure", model="ignored-model-name")
    print("Azure Chat Response:", response)


def _test_openai_embed():
    response = request_to_embed(["Hello", "World"], model="text-embedding-3-small", provider="openai")
    print("OpenAI Embedding Response (length):", len(response), len(response[0]) if response else 0)

def _test_azure_embed():
     if not use_azure:
        print("Skipping Azure embedding test as USE_AZURE is not true.")
        return
     # Azureの場合、model引数は無視され、環境変数のデプロイ名が使われる
     response = request_to_embed(["こんにちは", "世界"], model="ignored-model-name", provider="azure")
     print("Azure Embedding Response (length):", len(response), len(response[0]) if response else 0)

def _test_local_embed():
    print("Testing local sentence-transformer embedding...")
    response = request_to_embed(["これはローカル埋め込みのテストです。"], model="", is_embedded_at_local=True)
    print("Local Embedding Response (length):", len(response), len(response[0]) if response else 0)

def _test_local_llm_chat():
     messages = [
         {"role": "system", "content": "あなたは親切なアシスタントです。"},
         {"role": "user", "content": "日本の首都はどこですか？"},
     ]
     try:
         # poetry run python -m services.llm などで実行する場合、
         # --local-llm-address localhost:1234 のように引数で渡すか、
         # 環境変数 LOCAL_LLM_ADDRESS=localhost:1234 を設定する
         # あるいは、デフォルトの localhost:11434 を使う
         address = os.getenv("LOCAL_LLM_ADDRESS", "localhost:11434")
         # モデル名はローカルサーバーで利用可能なものを指定
         # 例: ollama run llama3 などで起動しているモデル
         response = request_to_chat_llm(messages=messages, provider="local", model="llama3", local_llm_address=address)
         print("Local LLM Chat Response:", response)
     except ConnectionError as e:
         print(f"Could not connect to local LLM: {e}")
     except Exception as e:
         print(f"Error during local LLM chat test: {e}")

def _test_local_llm_embed():
    texts = ["ローカルLLMでの埋め込みテストです。", "これは別の文章です。"]
    try:
        address = os.getenv("LOCAL_LLM_ADDRESS", "localhost:11434")
        # 埋め込みモデル名を指定 (例: nomic-embed-text)
        model = "nomic-embed-text" # Ollamaでよく使われるモデル例
        response = request_to_embed(texts, model=model, provider="local", local_llm_address=address)
        print("Local LLM Embedding Response (length):", len(response), len(response[0]) if response else 0)
    except ConnectionError as e:
         print(f"Could not connect to local LLM for embedding: {e}")
    except Exception as e:
         print(f"Error during local LLM embedding test: {e}")


if __name__ == "__main__":
    print("Running LLM service tests...")
    # _test_openai_chat()
    # _test_azure_chat()
    # _test_openai_embed()
    # _test_azure_embed()
    # _test_local_embed()
    _test_local_llm_chat()
    # _test_local_llm_embed()
    print("LLM service tests finished.")