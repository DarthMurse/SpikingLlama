from datasets import load_dataset
import os 
import random
from tqdm import tqdm
from pathlib import Path
import numpy as np
from multiprocessing import Process, cpu_count

import src.packed_dataset as packed_dataset
from src import Tokenizer

def prepare_full(tokenizer_path: Path,
                 destination_path: Path, 
                 chunk_size: int, 
                 split="train", 
                 text_list=None,
                 process_id: int = 0) -> None:
    destination_path.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(tokenizer_path)

    builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=f"{split}_wikitext_{process_id}",
            chunk_size=chunk_size,
            sep_token=tokenizer.bos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size
            )

    for text in tqdm(text_list):
        text = text["text"]
        text_ids = tokenizer.encode(text)
        builder.add_array(np.array(text_ids, dtype=builder.dtype))

def prepare(
        tokenizer_path: Path = Path("checkpoints/"),
        destination_path: Path = Path("data/wikitext_processed"),
        chunk_size: int = 2049 * 1024
        ) -> None:
    all_set = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    train_set = list(all_set["train"])
    val_set = list(all_set["validation"])
    #print(dataset[0])
    print("start")

    prepare_full(tokenizer_path, destination_path, chunk_size, "train", train_set)
    print("finished")

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)
