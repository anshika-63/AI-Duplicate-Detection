from datasets import load_dataset


# Load duplicate metadata
duplicate_dataset = load_dataset(
    "stefan-hf/rvlcdip-duplicates",
    "documents",
    split="test"
)

# Get the first document that has a known duplicate
duplicate_document = next(
    row for row in duplicate_dataset
    if row["has_train_duplicate"] is True
)

target_id = duplicate_document["id"]

print("Target document ID:")
print(target_id)

print("\nKnown duplicate:")
print(duplicate_document["train_duplicates"])


# Load OCR data
ocr_dataset = load_dataset(
    "stefan-hf/rvlcdip-redact",
    "textract_redact",
    split="test",
    streaming=True
)


# Find matching OCR document
for row in ocr_dataset:
    if row["id"] == target_id:
        print("\nOCR document found!")
        print("Document ID:")
        print(row["id"])

        print("\nLabel:")
        print(row["label"])

        print("\nOCR Text:")
        print(row["text"])

        break
else:
    print("\nOCR document not found.")