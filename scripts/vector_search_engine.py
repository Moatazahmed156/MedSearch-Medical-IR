"""
Vector Space Model Search Engine with Cosine Similarity
Supports TF-IDF ranking with word, phrase, and boolean search
"""

import json
import os
import math
import re
from collections import defaultdict

# Configuration
INDEX_PATH = "../data/inverted_index.json"
POSITIONAL_INDEX_PATH = "../data/positional_index.json"
PROCESSED_DOCS_DIR = "../data/processed_documents"
RAW_DOCS_DIR = "../data/raw_documents"


class VectorSearchEngine:
    """Search engine using Vector Space Model with TF-IDF and Cosine Similarity"""

    def __init__(self):
        print("🔧 Initializing Vector Search Engine...")

        # Load indexes
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            self.inverted_index = json.load(f)

        with open(POSITIONAL_INDEX_PATH, "r", encoding="utf-8") as f:
            self.positional_index = json.load(f)

        # Count documents from processed directory
        self.N = len([f for f in os.listdir(PROCESSED_DOCS_DIR) if f.endswith('.json')])
        
        # FIX 1: Handle empty document collection
        if self.N == 0:
            raise ValueError("No documents found in processed documents directory")

        # Calculate IDF for all terms
        self.idf = self._calculate_idf()

        # Build document vectors from processed JSON
        self.doc_vectors = self._build_document_vectors()

        print(f"✅ Loaded {len(self.doc_vectors)} document vectors")
        print(f"📊 Vocabulary size: {len(self.inverted_index)} terms\n")

    def _calculate_idf(self):
        """Calculate Inverse Document Frequency for all terms"""
        idf = {}
        for term, docs in self.inverted_index.items():
            df = len(docs)  # document frequency
            # FIX 2: Add small epsilon to prevent log(0) and improve numerical stability
            idf[term] = math.log((self.N + 1) / (df + 1))
        return idf

    def _build_document_vectors(self):
        """Build TF-IDF vectors from processed documents"""
        doc_vectors = {}
        print("📦 Building document vectors...")

        for filename in os.listdir(PROCESSED_DOCS_DIR):
            if filename.endswith(".json"):
                doc_id = filename.replace(".json", "")
                
                with open(os.path.join(PROCESSED_DOCS_DIR, filename), "r", encoding="utf-8") as f:
                    data = json.load(f)
                    text = data.get("clean_text", "")
                    words = text.split()

                # Calculate term frequency
                tf_counts = defaultdict(int)
                for word in words:
                    if word:  # skip empty strings
                        tf_counts[word] += 1

                # Calculate TF-IDF
                tfidf = {}
                for term, tf in tf_counts.items():
                    tfidf[term] = tf * self.idf.get(term, 0)
                
                doc_vectors[doc_id] = tfidf

        return doc_vectors

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        # FIX 3: Handle empty vectors
        if not vec1 or not vec2:
            return 0.0
            
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum(vec1[t] * vec2[t] for t in intersection)
        
        mag1 = math.sqrt(sum(v * v for v in vec1.values()))
        mag2 = math.sqrt(sum(v * v for v in vec2.values()))
        
        # FIX 4: Add epsilon to prevent division by zero
        denominator = mag1 * mag2
        return (numerator / denominator) if denominator > 1e-10 else 0.0

    def _build_query_vector(self, query_terms):
        """Build TF-IDF vector for query"""
        tf = defaultdict(int)
        for term in query_terms:
            if term:  # skip empty terms
                tf[term] += 1
        
        return {t: count * self.idf.get(t, 0) for t, count in tf.items()}

    def _phrase_search(self, phrase):
        """
        Search for exact phrase using positional index
        Returns set of matching document IDs
        """
        phrase = phrase.lower().strip()
        words = phrase.split()
        
        if not words:
            return set()
        
        # FIX 5: Handle single-word phrases
        if len(words) == 1:
            word = words[0]
            if word in self.positional_index:
                return set(self.positional_index[word].keys())
            return set()
        
        # Get documents containing first word
        first_word = words[0]
        if first_word not in self.positional_index:
            return set()
        
        candidate_docs = set(self.positional_index[first_word].keys())
        matching_docs = set()

        # Check each candidate document
        for doc_id in candidate_docs:
            # Get positions of first word in this document
            first_positions = self.positional_index[first_word][doc_id]
            
            # Check if phrase exists starting from each position
            for start_pos in first_positions:
                phrase_found = True
                
                # Check if subsequent words appear in correct positions
                for i, word in enumerate(words[1:], 1):
                    expected_pos = start_pos + i
                    
                    # Check if word exists in document at expected position
                    if word not in self.positional_index or \
                       doc_id not in self.positional_index[word] or \
                       expected_pos not in self.positional_index[word][doc_id]:
                        phrase_found = False
                        break
                
                if phrase_found:
                    matching_docs.add(doc_id)
                    break  # Found phrase in this doc, no need to check more positions

        return matching_docs

    def _boolean_filter(self, query):
        """
        Advanced boolean filter:
        Supports multiple AND / OR / NOT (left to right)
        """
        tokens = query.lower().split()

        if not any(t in {"and", "or", "not"} for t in tokens):
            return None

        result_docs = None
        current_op = None

        i = 0
        while i < len(tokens):
            token = tokens[i]

            if token in {"and", "or", "not"}:
                current_op = token
                i += 1
                continue

            # token = term
            term_docs = set(self.inverted_index.get(token, {}).keys())

            if result_docs is None:
                result_docs = term_docs

            else:
                if current_op == "and":
                    result_docs &= term_docs
                elif current_op == "or":
                    result_docs |= term_docs
                elif current_op == "not":
                    result_docs -= term_docs

            i += 1

        # FIX 6: Return empty set instead of None if no results
        return result_docs if result_docs is not None else set()

    def search(self, query, top_k=10):
        """
        Main search function supporting:
        - Word search: diabetes
        - Phrase search: "diabetes mellitus"
        - Boolean search: diabetes AND treatment, cancer OR tumor, heart NOT failure
        - Combined: "breast cancer" AND treatment
        """
        original_query = query
        query_lower = query.lower()
        filtered_docs = None

        # 1. Handle phrase search (text in quotes)
        if '"' in query:
            phrases = re.findall(r'"([^"]+)"', query)
            for phrase in phrases:
                phrase_docs = self._phrase_search(phrase)
                filtered_docs = phrase_docs if filtered_docs is None else filtered_docs & phrase_docs
            
            # Remove phrases from query for further processing
            query_lower = re.sub(r'"[^"]+"', '', query_lower).strip()

        # 2. Handle boolean operators
        boolean_docs = self._boolean_filter(query_lower)
        if boolean_docs is not None:
            filtered_docs = boolean_docs if filtered_docs is None else filtered_docs & boolean_docs
            # Remove boolean operators for vector search
            query_lower = re.sub(r'\b(and|or|not)\b', '', query_lower, flags=re.I).strip()

        # 3. Vector search on remaining terms
        query_terms = query_lower.split()
        
        # FIX 7: Handle empty query after cleanup
        if not query_terms and filtered_docs is not None:
            # If we have filtered docs but no query terms, return filtered docs
            results = []
            for doc_id in list(filtered_docs)[:top_k]:
                snippet = self._get_snippet(doc_id, original_query)
                results.append((doc_id, 1.0, snippet))
            return results
        
        query_vector = self._build_query_vector(query_terms)

        # Determine which documents to score
        docs_to_check = filtered_docs if filtered_docs is not None else self.doc_vectors.keys()
        scores = {}

        for doc_id in docs_to_check:
            similarity = self._cosine_similarity(self.doc_vectors.get(doc_id, {}), query_vector)
            if similarity > 0:
                scores[doc_id] = similarity

        # Sort by score (descending)
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Get results with snippets
        results = []
        for doc_id, score in ranked_docs[:top_k]:
            snippet = self._get_snippet(doc_id, original_query)
            results.append((doc_id, score, snippet))

        return results

    def _get_snippet(self, doc_id, query, max_len=200):
        """Extract relevant snippet from document"""
        try:
            # Try to get original document
            raw_path = os.path.join(RAW_DOCS_DIR, f"{doc_id}.json")
            
            if os.path.exists(raw_path):
                with open(raw_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Combine title and abstract/summary for snippet
                    text = data.get("title", "") + " " + data.get("abstract", data.get("summary", ""))
            else:
                # Fallback to processed document
                proc_path = os.path.join(PROCESSED_DOCS_DIR, f"{doc_id}.json")
                with open(proc_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    text = data.get("clean_text", "")

            # FIX 8: Handle empty text
            if not text.strip():
                return "[No text available]"

            # Find best snippet containing query terms
            query_clean = re.sub(r'["\']', '', query)
            query_clean = re.sub(r'\b(and|or|not)\b', '', query_clean, flags=re.I)
            terms = query_clean.lower().split()

            best_pos = 0
            text_lower = text.lower()

            # Find position of first query term
            for term in terms:
                pos = text_lower.find(term)
                if pos >= 0:
                    best_pos = pos
                    break

            # Extract snippet around the found term
            start = max(0, best_pos - 50)
            end = min(len(text), best_pos + max_len)
            snippet = text[start:end].strip()

            # Add ellipsis if needed
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet += "..."

            # FIX 9: Ensure we return something meaningful
            return snippet if snippet else text[:max_len] + "..."

        except Exception as e:
            return f"[Error getting snippet: {e}]"

    def display_results(self, query, results):
        """Display search results in a readable format"""
        print(f"\n{'='*80}")
        print(f"🔎 Query: {query}")
        print(f"📊 Found {len(results)} results")
        print(f"{'='*80}\n")

        if not results:
            print("❌ No matching documents found.\n")
            print("💡 Tips:")
            print("   - Try different keywords")
            print("   - Use phrase search: \"exact phrase\"")
            print("   - Use boolean: diabetes AND treatment")
            return

        for i, (doc_id, score, snippet) in enumerate(results, 1):
            print(f"{i}. Document ID: {doc_id}")
            print(f"   📈 Relevance Score: {score:.4f}")
            print(f"   📄 Snippet: {snippet}")
            print()

        print(f"{'='*80}\n")


def main():
    """Interactive search interface"""
    try:
        engine = VectorSearchEngine()
        
        print("=" * 60)
        print("🔍 Medical Document Search Engine")
        print("=" * 60)
        print("\n💡 Search Tips:")
        print("   • Word search: diabetes")
        print("   • Phrase search: \"diabetes mellitus\"")
        print("   • Boolean AND: diabetes AND treatment")
        print("   • Boolean OR: cancer OR tumor")
        print("   • Boolean NOT: heart NOT failure")
        print("   • Combined: \"breast cancer\" AND treatment")
        print("\n   Type 'exit' or 'quit' to stop\n")
        
        while True:
            query = input("🔍 Enter your search query: ").strip()
            
            if query.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                print("⚠️  Please enter a search query\n")
                continue
            
            results = engine.search(query, top_k=10)
            engine.display_results(query, results)
    
    except FileNotFoundError as e:
        print(f"\n❌ Error: Required index files not found!")
        print(f"   {e}")
        print("\n💡 Please run these commands first:")
        print("   1. python preprocessing.py")
        print("   2. python build_index.py")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()