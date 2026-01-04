"""
Document Preprocessing for Search Engine
Processes JSON documents while preserving structure
Extracts textual fields and preprocesses them for indexing
"""

import os
import re
import json

# Configuration
RAW_DOCS_DIR = "../data/raw_documents"
CLEAN_DOCS_DIR = "../data/processed_documents"

TEXT_FIELDS = ["title", "summary", "abstract"]  # fields to process


def normalize_text(text):
    """
    Normalize text while preserving structure for phrase search
    - lowercase
    - remove non-text characters except spaces
    - clean spacing
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    # Keep only letters and spaces (important for phrase search)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Clean multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_text_field(text):
    """Normalize text for indexing"""
    return normalize_text(text)


def preprocess_document_json(data):
    """
    Preprocess text fields in JSON
    Returns updated JSON with 'clean_text' for indexing
    """
    processed_text_parts = []

    # Process each text field
    for key in TEXT_FIELDS:
        if key in data and isinstance(data[key], str):
            processed = preprocess_text_field(data[key])
            if processed:  # only add non-empty text
                processed_text_parts.append(processed)

    # Join all processed text into one searchable block
    # This is the key field used by the indexer
    data["clean_text"] = " ".join(processed_text_parts)

    return data


def process_all_documents():
    """Process all JSON documents in RAW_DOCS_DIR"""
    os.makedirs(CLEAN_DOCS_DIR, exist_ok=True)

    doc_files = [f for f in os.listdir(RAW_DOCS_DIR) if f.endswith(".json")]
    total_docs = len(doc_files)
    processed_count = 0

    print(f"📄 Found {total_docs} JSON documents")
    print("🔧 Processing textual fields for search indexing...\n")

    for filename in doc_files:
        try:
            input_path = os.path.join(RAW_DOCS_DIR, filename)
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Preprocess JSON content
            updated_data = preprocess_document_json(data)

            # Save processed JSON
            output_path = os.path.join(CLEAN_DOCS_DIR, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(updated_data, f, ensure_ascii=False, indent=4)

            processed_count += 1

            if processed_count % 100 == 0:
                print(f"✓ Processed {processed_count}/{total_docs} documents")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    print(f"\n✅ Successfully processed {processed_count}/{total_docs} documents!")
    print(f"📁 Output directory: {os.path.abspath(CLEAN_DOCS_DIR)}")
    print(f"\n💡 Next step: Run build_index.py to create search indexes")


if __name__ == "__main__":
    process_all_documents()