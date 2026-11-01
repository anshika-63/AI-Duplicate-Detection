from datasets import load_dataset


dataset = load_dataset(
    "stefan-hf/rvlcdip-duplicates",
    "documents",
    split="test"
)

duplicate_documents = dataset.filter(
    lambda row: row["has_train_duplicate"] is True
).select(range(10))

non_duplicate_documents = dataset.filter(
    lambda row: row["has_train_duplicate"] is False
).select(range(10))

print("Duplicate documents:", len(duplicate_documents))
print("Non-duplicate documents:", len(non_duplicate_documents))

print("\nFirst duplicate:")
print(duplicate_documents[0])

print("\nFirst non-duplicate:")
print(non_duplicate_documents[0])