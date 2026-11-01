from datasets import load_dataset

dataset = load_dataset(
    "stefan-hf/rvlcdip-duplicates",
    "documents",
    split="test"
)

print(dataset)
print(dataset[0])