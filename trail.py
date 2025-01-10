from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
train_set = dataset["train"]
print(train_set['text'])
