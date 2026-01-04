"""
Flask Web Application for Medical Document Search Engine
Features: Vector Space Model, Phrase Search, Boolean Search, Pagination
Works with JSON documents and supports all search types
"""
from flask import Flask, render_template, request, Response, jsonify
import os
import json
import math
import re
from collections import defaultdict


# Flask app initialization
app = Flask(__name__)


# ===== Configuration =====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DOCS_DIR = os.path.join(BASE_DIR, "data", "processed_documents")
RAW_DOCS_DIR = os.path.join(BASE_DIR, "data", "raw_documents")
INDEX_PATH = os.path.join(BASE_DIR, "data", "inverted_index.json")
POSITIONAL_INDEX_PATH = os.path.join(BASE_DIR, "data", "positional_index.json")


class WebSearchEngine:
    """Search engine for web interface with all search capabilities"""
    
    def __init__(self):
        """Initialize and load all necessary data"""
        print("🔧 Loading search engine for web interface...")
        
        # Load processed documents (JSON format with clean_text)
        self.documents = {}
        for filename in os.listdir(PROCESSED_DOCS_DIR):
            if filename.endswith(".json"):
                doc_id = filename.replace(".json", "")
                try:
                    with open(os.path.join(PROCESSED_DOCS_DIR, filename), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.documents[doc_id] = data.get("clean_text", "")
                except Exception as e:
                    print(f"⚠️  Error loading {filename}: {e}")
        
        # Load raw documents for display (JSON format with original fields)
        self.raw_documents = {}
        for filename in os.listdir(RAW_DOCS_DIR):
            if filename.endswith(".json"):
                doc_id = filename.replace(".json", "")
                try:
                    with open(os.path.join(RAW_DOCS_DIR, filename), "r", encoding="utf-8") as f:
                        self.raw_documents[doc_id] = json.load(f)
                except Exception as e:
                    print(f"⚠️  Error loading raw {filename}: {e}")
        
        # Load inverted index
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            self.inverted_index = json.load(f)
        
        # Load positional index for phrase search
        with open(POSITIONAL_INDEX_PATH, "r", encoding="utf-8") as f:
            self.positional_index = json.load(f)
        
        # Calculate IDF
        self.N = len(self.documents)
        self.idf = self._calculate_idf()
        
        # Build document vectors
        self.doc_vectors = self._build_document_vectors()
        
        print(f"✅ Loaded {len(self.documents)} documents")
        print(f"✅ Loaded {len(self.raw_documents)} raw documents")
        print(f"✅ Vocabulary: {len(self.inverted_index)} terms")
    
    def _calculate_idf(self):
        """Calculate IDF for all terms"""
        idf = {}
        for term, docs in self.inverted_index.items():
            df = len(docs)
            idf[term] = math.log(self.N / df) if df > 0 else 0
        return idf
    
    def _build_document_vectors(self):
        """Build TF-IDF vectors for all documents"""
        doc_vectors = {}
        
        for doc_id, text in self.documents.items():
            words = text.split()
            word_counts = defaultdict(int)
            
            for word in words:
                if word:
                    word_counts[word] += 1
            
            tfidf = {}
            for word, tf in word_counts.items():
                tfidf[word] = tf * self.idf.get(word, 0)
            
            doc_vectors[doc_id] = tfidf
        
        return doc_vectors
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum(vec1[x] * vec2[x] for x in intersection)
        
        sum1 = sum(v**2 for v in vec1.values())
        sum2 = sum(v**2 for v in vec2.values())
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _phrase_search(self, phrase):
        """Search for exact phrase using positional index"""
        phrase = phrase.lower().strip()
        words = phrase.split()
        
        if not words or words[0] not in self.positional_index:
            return set()
        
        candidate_docs = set(self.positional_index[words[0]].keys())
        matching_docs = set()
        
        for doc_id in candidate_docs:
            positions = self.positional_index[words[0]][doc_id]
            
            for start_pos in positions:
                phrase_found = True
                
                for i, word in enumerate(words[1:], start=1):
                    expected_pos = start_pos + i
                    
                    if word not in self.positional_index or \
                       doc_id not in self.positional_index[word] or \
                       expected_pos not in self.positional_index[word][doc_id]:
                        phrase_found = False
                        break
                
                if phrase_found:
                    matching_docs.add(doc_id)
                    break
        
        return matching_docs
    
    def _boolean_filter(self, query):
        """Handle boolean operators: AND, OR, NOT"""
        query_upper = query.upper()

        # AND operator
        if " AND " in query_upper:
            terms = re.split(r'\s+AND\s+', query_upper)
            docs = set(self.documents.keys())
            
            for term in terms:
                term_lower = term.strip().lower()
                term_docs = set(self.inverted_index.get(term_lower, {}).keys())
                docs &= term_docs
            
            return docs

        # OR operator
        if " OR " in query_upper:
            terms = re.split(r'\s+OR\s+', query_upper)
            docs = set()
            
            for term in terms:
                term_lower = term.strip().lower()
                term_docs = set(self.inverted_index.get(term_lower, {}).keys())
                docs |= term_docs
            
            return docs

        # NOT operator
        if " NOT " in query_upper:
            parts = re.split(r'\s+NOT\s+', query_upper, maxsplit=1)
            if len(parts) == 2:
                include_term = parts[0].strip().lower()
                exclude_term = parts[1].strip().lower()
                
                include_docs = set(self.inverted_index.get(include_term, {}).keys())
                exclude_docs = set(self.inverted_index.get(exclude_term, {}).keys())
                
                return include_docs - exclude_docs

        return None
    
    def search(self, query):
        """
        Main search function with support for:
        - Word search: diabetes
        - Phrase search: "diabetes mellitus"
        - Boolean AND: diabetes AND treatment
        - Boolean OR: cancer OR tumor
        - Boolean NOT: heart NOT failure
        - Combined: "breast cancer" AND treatment
        
        Returns ALL matching results (no limit)
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
            
            # Remove phrases from query
            query_lower = re.sub(r'"[^"]+"', '', query_lower).strip()

        # 2. Handle boolean operators
        boolean_docs = self._boolean_filter(query_lower)
        if boolean_docs is not None:
            filtered_docs = boolean_docs if filtered_docs is None else filtered_docs & boolean_docs
            # Remove boolean operators for vector search
            query_lower = re.sub(r'\b(and|or|not)\b', '', query_lower, flags=re.I).strip()

        # 3. Vector search on remaining terms
        query_terms = query_lower.split()
        query_vector = self._build_query_vector(query_terms)

        # Determine which documents to score
        docs_to_check = filtered_docs if filtered_docs else self.doc_vectors.keys()
        scores = {}

        for doc_id in docs_to_check:
            similarity = self._cosine_similarity(self.doc_vectors.get(doc_id, {}), query_vector)
            if similarity > 0:
                scores[doc_id] = similarity

        # Sort by score (descending)
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked_docs
    
    def _build_query_vector(self, query_terms):
        """Build TF-IDF vector for query"""
        tf = defaultdict(int)
        for term in query_terms:
            if term:
                tf[term] += 1
        
        return {t: count * self.idf.get(t, 0) for t, count in tf.items()}
    
    def get_snippet(self, doc_id, query, max_chars=300):
        """Get highlighted snippet from document"""
        if doc_id not in self.raw_documents:
            return "[Document not found]"
        
        doc_data = self.raw_documents[doc_id]
        
        # Combine title, summary, and abstract for snippet
        text_parts = []
        if "title" in doc_data:
            text_parts.append(doc_data["title"])
        if "summary" in doc_data:
            text_parts.append(doc_data["summary"])
        elif "abstract" in doc_data:
            text_parts.append(doc_data["abstract"])
        
        text = " ".join(text_parts)
        
        # Extract query words
        query_clean = re.sub(r'["\']', '', query)
        query_clean = re.sub(r'\b(and|or|not)\b', '', query_clean, flags=re.I)
        query_words = set(w.lower() for w in query_clean.split() if w)
        
        # Find best snippet position
        text_lower = text.lower()
        best_pos = 0
        
        for word in query_words:
            pos = text_lower.find(word)
            if pos >= 0:
                best_pos = pos
                break
        
        # Extract snippet
        start = max(0, best_pos - 100)
        end = min(len(text), best_pos + max_chars)
        snippet = text[start:end]
        
        # Highlight query words
        for word in query_words:
            pattern = re.compile(r'\b(' + re.escape(word) + r')\b', re.IGNORECASE)
            snippet = pattern.sub(r"<span class='highlight'>\1</span>", snippet)
        
        # Add ellipsis
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet += "..."
        
        return snippet
    
    def get_full_document(self, doc_id):
        """Get full raw document content formatted for display"""
        if doc_id not in self.raw_documents:
            return "Document not found"
        
        doc_data = self.raw_documents[doc_id]
        
        # Format document for display
        formatted = []
        
        if "title" in doc_data:
            formatted.append(f"TITLE:\n{doc_data['title']}\n")
        
        if "doc_id" in doc_data:
            formatted.append(f"DOCUMENT ID:\n{doc_data['doc_id']}\n")
        
        if "writer" in doc_data:
            formatted.append(f"AUTHOR:\n{doc_data['writer']}\n")
        
        if "year" in doc_data:
            formatted.append(f"YEAR:\n{doc_data['year']}\n")
        
        if "link" in doc_data:
            formatted.append(f"SOURCE:\n{doc_data['link']}\n")
        
        if "abstract" in doc_data:
            formatted.append(f"ABSTRACT:\n{doc_data['abstract']}\n")
        elif "summary" in doc_data:
            formatted.append(f"SUMMARY:\n{doc_data['summary']}\n")
        
        return "\n".join(formatted)
    
    def get_document_metadata(self, doc_id):
        """Get document metadata as dictionary"""
        if doc_id not in self.raw_documents:
            return None
        return self.raw_documents[doc_id]


# Initialize search engine
print("🚀 Initializing Flask application...")
try:
    search_engine = WebSearchEngine()
    print("✅ Search engine initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing search engine: {e}")
    print("💡 Make sure you've run preprocessing.py and build_index.py first")
    raise


# ===== Flask Routes =====

@app.route("/", methods=["GET", "POST"])
def home():
    """Main search page with pagination"""
    results = []
    snippets = {}
    query = ""
    search_time = 0
    page = 1
    total_pages = 0
    total_results = 0
    per_page = 20
    
    # Get page number and query
    if request.method == "GET":
        page = int(request.args.get('page', 1))
        query = request.args.get('query', '').strip()
    else:  # POST
        query = request.form.get("query", "").strip()
        page = 1  # Reset to first page on new search
    
    if query:
        import time
        start_time = time.time()
        
        try:
            # Perform search - GET ALL RESULTS (no limit)
            all_results = search_engine.search(query)
            
            # Calculate pagination
            total_results = len(all_results)
            total_pages = (total_results + per_page - 1) // per_page if total_results > 0 else 1
            
            # Ensure page is within valid range
            page = max(1, min(page, total_pages))
            
            # Get results for current page
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            results = all_results[start_idx:end_idx]
            
            # Get snippets for current page only
            for doc_id, score in results:
                snippets[doc_id] = search_engine.get_snippet(doc_id, query)
            
            search_time = round((time.time() - start_time) * 1000, 2)
        
        except Exception as e:
            print(f"❌ Search error: {e}")
            search_time = 0
    
    return render_template(
        "index.html",
        results=results,
        snippets=snippets,
        query=query,
        search_time=search_time,
        page=page,
        total_pages=total_pages,
        total_results=total_results,
        per_page=per_page
    )


@app.route("/full_document")
def full_document():
    """View full document with formatted content"""
    doc_id = request.args.get("doc_id")
    query = request.args.get("query", "")
    
    if not doc_id:
        return "<h2>Error: No document ID provided</h2>", 400
    
    metadata = search_engine.get_document_metadata(doc_id)
    
    return render_template(
        "full_document.html",
        doc_id=doc_id,
        metadata=metadata,
        query=query
    )


@app.route("/api/search")
def api_search():
    """JSON API endpoint for search"""
    query = request.args.get("query", "").strip()
    limit = int(request.args.get("limit", 10))
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        results = search_engine.search(query)[:limit]
        
        response = {
            "query": query,
            "total_results": len(results),
            "results": [
                {
                    "doc_id": doc_id,
                    "score": score,
                    "snippet": search_engine.get_snippet(doc_id, query)
                }
                for doc_id, score in results
            ]
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== Error Handlers =====

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template("error.html", error="Page not found (404)"), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template("error.html", error="Internal server error (500)"), 500


# ===== Run Application =====

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🏥 MEDICAL DOCUMENT SEARCH ENGINE - WEB INTERFACE")
    print("=" * 70)
    print(f"📊 Loaded documents: {len(search_engine.documents)}")
    print(f"📚 Vocabulary size: {len(search_engine.inverted_index)} terms")
    print(f"🌐 Server URL: http://localhost:5000")
    print(f"🔍 API endpoint: http://localhost:5000/api/search?query=diabetes")
    print("=" * 70)
    print("\n💡 Search Examples:")
    print("   • Word search: diabetes")
    print("   • Phrase search: \"diabetes mellitus\"")
    print("   • Boolean AND: diabetes AND treatment")
    print("   • Boolean OR: cancer OR tumor")
    print("   • Boolean NOT: heart NOT failure")
    print("   • Combined: \"breast cancer\" AND treatment")
    print("\n" + "=" * 70 + "\n")
    
    app.run(debug=True, host="0.0.0.0", port=5000)