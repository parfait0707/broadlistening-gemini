import pandas as pd
from tqdm import tqdm

from services.llm import request_to_embed


def embedding(config):
    model = config["embedding"]["model"]

    dataset = config["output_dir"]
    path = f"outputs/{dataset}/embeddings.pkl"
    arguments = pd.read_csv(f"outputs/{dataset}/args.csv", usecols=["arg-id", "argument"])
    embeddings = []
    batch_size = 1000
    for i in tqdm(range(0, len(arguments), batch_size)):
        args = arguments["argument"].tolist()[i : i + batch_size]
        embeds = request_to_embed(args, model)
        embeddings.extend(embeds)
    df = pd.DataFrame([{"arg-id": arguments.iloc[i]["arg-id"], "embedding": e} for i, e in enumerate(embeddings)])
    df.to_pickle(path)