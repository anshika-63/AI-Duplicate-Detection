import hashlib
import csv
from pathlib import Path

DATASET_FOLDER = Path("dataset")
REPORT_FILE = "duplicate_report.csv"  


def calculate_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def find_duplicate():
    hash_group = {}
    for file_path in DATASET_FOLDER.rglob("*"):
        if file_path.is_file():
            file_hash = calculate_hash(file_path)
            if file_hash not in hash_group:
                hash_group[file_hash] = []
            hash_group[file_hash].append(file_path)
    duplicates = {file_hash: files
                  for file_hash, files in hash_group.items() if len(files) > 1}

    return duplicates


def export_report(duplicates):
    with open(REPORT_FILE, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(["SHA-256 HASH", "File Path"])

        for file_hash, files in duplicates.items():
            for file_path in files:
                writer.writerow([file_hash, file_path])

    print(f"\nReport generated: {REPORT_FILE}")

if __name__ == "__main__":
    duplicates = find_duplicate()
    if duplicates:
        print("\nExact Duplicates files Found:\n")

        for file_hash, files in duplicates.items():
            print("Hash:", file_hash)

            for file_path in files:
                print(" -", file_path)

            print()
    else:
        print("\nNo duplicates found.")

    export_report(duplicates)
