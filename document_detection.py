from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):

    text = text.lower()
    text = " ".join(text.split())
    return text


def calculate_tfidf_similarity(text1 , text2):

    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    documents = [text1, text2]

    vectorizer = TfidfVectorizer()

    tfidf_vector = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(tfidf_vector[0:1], 
    tfidf_vector[1:2])[0][0]

    return similarity*100

def get_duplicate_documents():

    duplicate_dataset = load_dataset("stefan-hf/rvlcdip-duplicates","documents",split ="test")

    for row in duplicate_dataset:
        if row["has_train_duplicate"]:
            return row
    return None


def get_ocr_text(target_id, split):

    ocr_dataset = load_dataset("stefan-hf/rvlcdip-redact","textract_redact",
     split=split, streaming=True)

    for row in ocr_dataset:
        if row["id"] == target_id:
            return row["text"]
    return None
        

if __name__ == "__main__":

    print("loading known duplicate pair...")

    duplicate_document = get_duplicate_documents()

    if duplicate_document is None:

        print("No Duplicate document found in the dataset.")
        exit()


    test_id = duplicate_document["id"]
    train_id = duplicate_document["train_duplicates"][0]

    print("\nTest document:")
    print(test_id)

    print("\nKnown training duplicate:")
    print(train_id)

    print("\nLoading OCR text for test document...")

    # test_text = get_ocr_text(
    #     test_id,
    #     "test"
    # )

    print("Test OCR loaded.")

    print("\nLoading OCR text for training duplicate...")

    train_text = get_ocr_text(
        train_id,
        "train"
    )

    print("Training OCR loaded.")

    # if test_text is None or train_text is None:

    #     print("\nCould not find OCR text.")
    #     exit()

    # similarity = calculate_tfidf_similarity(
    #     test_text,
    #     train_text
    # )

    print("\n--------------------------------")
    print("DOCUMENT SIMILARITY RESULT")
    print("--------------------------------")

    print(f"Test document: {test_id}")
    print(f"Training document: {train_id}")

    # print(f"\nTF-IDF Similarity: {similarity:.2f}%")
    

