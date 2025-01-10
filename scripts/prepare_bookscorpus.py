import json
import glob
import os
from pathlib import Path
import sys
from typing import List
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import src.packed_dataset as packed_dataset
from src import Tokenizer

def prepare_full(
    source_path: Path = Path('data/'),
    tokenizer_path: Path = Path('checkpoints/'),
    destination_path: Path = Path('data/bookcorpus'),
    chunk_size: int = 2049 * 1024,
    split: str="train",
) -> None:
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_bookcorpus",  # Use process_id to differentiate builders
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    print("processing p1-"+split)
    with open(source_path / "books_large_p1.txt", "r") as f:
        para = f.readlines()
        if split == "train":
            para = para[:int(0.85*len(para))]
        if split == "valid":
            para = para[int(0.85*len(para)):]
        for i in tqdm(range(len(para))):
            text = para[i]
            text_ids = tokenizer.encode(text)
            builder.add_array(np.array(text_ids, dtype=builder.dtype))

    print("processing p2-"+split)
    with open(source_path / "books_large_p2.txt", "r") as f:
        para = f.readlines()
        if split == "train":
            para = para[:int(0.85*len(para))]
        if split == "valid":
            para = para[int(0.85*len(para)):]
        for i in tqdm(range(len(para))):
            text = para[i]
            text_ids = tokenizer.encode(text)
            builder.add_array(np.array(text_ids, dtype=builder.dtype))

    # we throw away the final corpus to avoid meaningless corpus filled with bos_ids, see https://github.com/jzhang38/TinyLlama/issues/83 for more details
    # builder.write_reminder()

if __name__ == "__main__":
    #prepare_full(split="train") 
    prepare_full(split="valid")
