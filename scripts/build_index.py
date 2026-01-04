"""
Inverted Index Builder for Medical Document Search Engine
Creates positional inverted index to support phrase and boolean search
"""
import os
import json
from collections import defaultdict


# Configuration
PROCESSED_DOCS_DIR = "../data/processed_documents"
INVERTED_INDEX_PATH = "../data/inverted_index.json"
POSITIONAL_INDEX_PATH = "../data/positional_index.json"
DOC_METADATA_PATH = "../data/doc_metadata.json"


class InvertedIndexBuilder:
    """Builds inverted index with positional information for phrase search"""

    def __init__(self):
        # word -> {doc_id: frequency}
        self.inverted_index = defaultdict(dict)

        # word -> {doc_id: [positions]}
        self.positional_index = defaultdict(lambda: defaultdict(list))

        # doc_id -> metadata
        self.doc_metadata = {}

    def build_index(self, docs_directory):
        """Read processed documents and build indexes"""
        if not os.path.exists(docs_directory):
            raise FileNotFoundError(f"❌ Directory not found: {docs_directory}")

        doc_files = [f for f in os.listdir(docs_directory) if f.endswith(".json")]
        
        if not doc_files:
            raise ValueError(f"❌ No JSON files found in {docs_directory}")

        total_docs = len(doc_files)

        print(f"🔨 Building inverted index from {total_docs} documents...\n")

        successful_count = 0
        empty_count = 0

        for idx, filename in enumerate(doc_files, 1):
            doc_id = filename.replace(".json", "")
            file_path = os.path.join(docs_directory, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Get the preprocessed text
                text = data.get("clean_text", "")

                if not text or not text.strip():
                    empty_count += 1
                    print(f"⚠️  Warning: Empty text in {filename}")
                    continue

                self._index_document(doc_id, text)
                successful_count += 1

                if successful_count % 200 == 0 or idx == total_docs:
                    print(f"✓ Indexed {successful_count}/{total_docs} documents")

            except json.JSONDecodeError as e:
                print(f"❌ JSON error in {filename}: {e}")
            except Exception as e:
                print(f"❌ Error indexing {filename}: {e}")

        print(f"\n✅ Indexing complete!")
        print(f"   📄 Successfully indexed: {successful_count}")
        if empty_count > 0:
            print(f"   ⚠️  Skipped (empty): {empty_count}")
        print(f"   📊 Total unique terms: {len(self.inverted_index)}")

    def _index_document(self, doc_id, text):
        """Index a single document with positional information"""
        words = text.split()

        # Store document metadata
        self.doc_metadata[doc_id] = {
            "total_words": len(words)
        }

        # Build frequency and positional indexes
        for position, word in enumerate(words):
            if not word:  # skip empty strings
                continue

            # Frequency count (for TF-IDF calculation)
            if doc_id not in self.inverted_index[word]:
                self.inverted_index[word][doc_id] = 0
            self.inverted_index[word][doc_id] += 1

            # Positional list (for phrase search)
            self.positional_index[word][doc_id].append(position)

    def save_indexes(self):
        """Write all indices to disk"""
        print("\n💾 Saving index files...")

        # Create data directory if it doesn't exist
        data_dir = os.path.dirname(INVERTED_INDEX_PATH)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        # Save inverted index
        with open(INVERTED_INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(self.inverted_index, f, indent=2)

        # Save positional index
        with open(POSITIONAL_INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump(self.positional_index, f, indent=2)

        # Save document metadata
        with open(DOC_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.doc_metadata, f, indent=2)

        print(f"✓ Inverted index saved: {INVERTED_INDEX_PATH}")
        print(f"✓ Positional index saved: {POSITIONAL_INDEX_PATH}")
        print(f"✓ Document metadata saved: {DOC_METADATA_PATH}")

    def get_statistics(self):
        """Get index statistics"""
        total_docs = len(self.doc_metadata)
        
        if total_docs == 0:
            return {
                "total_documents": 0,
                "unique_terms": 0,
                "avg_doc_length": 0
            }

        avg_len = sum(meta["total_words"] for meta in self.doc_metadata.values()) / total_docs

        # Calculate some additional stats
        total_terms = sum(len(docs) for docs in self.inverted_index.values())

        return {
            "total_documents": total_docs,
            "unique_terms": len(self.inverted_index),
            "avg_doc_length": round(avg_len, 2),
            "total_postings": total_terms
        }

    def verify_index(self):
        """Verify index integrity"""
        print("\n🔍 Verifying index integrity...")
        
        issues = []

        # Check if indexes are consistent
        for word in self.inverted_index:
            if word not in self.positional_index:
                issues.append(f"Word '{word}' in inverted_index but not in positional_index")

        for word in self.positional_index:
            if word not in self.inverted_index:
                issues.append(f"Word '{word}' in positional_index but not in inverted_index")

        if issues:
            print(f"⚠️  Found {len(issues)} consistency issues:")
            for issue in issues[:5]:  # Show first 5 issues
                print(f"   - {issue}")
            if len(issues) > 5:
                print(f"   ... and {len(issues) - 5} more")
        else:
            print("✅ Index integrity verified - no issues found")


def main():
    """Main function to build all indexes"""
    print("=" * 70)
    print("🚀 Medical Document Search Engine - Index Builder")
    print("=" * 70)

    try:
        # Build indexes
        builder = InvertedIndexBuilder()
        builder.build_index(PROCESSED_DOCS_DIR)
        
        # Verify integrity
        builder.verify_index()
        
        # Save to disk
        builder.save_indexes()

        # Display statistics
        stats = builder.get_statistics()
        print("\n" + "=" * 70)
        print("📈 Index Statistics:")
        print(f"   📄 Total documents: {stats['total_documents']}")
        print(f"   📚 Unique terms: {stats['unique_terms']}")
        print(f"   📏 Average doc length: {stats['avg_doc_length']} words")
        print(f"   🔗 Total postings: {stats['total_postings']}")
        print("=" * 70)

        print("\n✅ Index building complete!")
        print("🎯 You can now run: python vector_search_engine.py")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure you've run preprocessing.py first!")
        print("   Run: python preprocessing.py")
    
    except ValueError as e:
        print(f"\n❌ Error: {e}")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()