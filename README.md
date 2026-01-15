# 🏥 MedSearch — Medical Information Retrieval System

MedSearch is a full-featured **Medical Information Retrieval (IR) system** designed to index, search, and rank medical research documents from **PubMed**.  
It implements a complete IR pipeline using a **Vector Space Model (VSM)** with **TF-IDF** and **Cosine Similarity** ranking, along with **Boolean retrieval** and **exact phrase search** using a positional index.

This project demonstrates real-world search engine architecture including **data acquisition, preprocessing, indexing, ranking, and a web-based search interface**.


## 🚀 Features

- 🔍 Keyword, Boolean (`AND`, `OR`, `NOT`) and phrase search (`"heart disease"`)
- 🧠 TF-IDF + Cosine Similarity ranking
- 🗂️ Inverted Index + Positional Index
- 🕷️ Data collection from PubMed (Scraping + NCBI Entrez API)
- 💻 Flask web interface with snippets, highlighting, pagination, and full document view
- ⚡ Handles **17,000+ documents** with fast query response


## 🏗️ Pipeline
 ```
 Data Collection → Preprocessing → Indexing → Search & Ranking → Web UI
 ```

- `generate_documents.py` / `generate_document_apis.py` → collect data  
- `preprocess_documents.py` → clean and normalize text  
- `build_index.py` → build inverted & positional indexes  
- `app.py` → search engine + Flask UI  

### 1️⃣ Data Acquisition

- `generate_documents.py`
  - Scrapes PubMed using BeautifulSoup and Requests
  - Extracts:
    - Title
    - PMID
    - Authors
    - Publication Year
    - Summary
    - Abstract

- `generate_document_apis.py`
  - Uses **NCBI Entrez (E-utils)** API
  - Fetches structured XML data for diseases such as:
    - Diabetes
    - Hypertension
    - COVID-19
  - Converts data to JSON

---

### 2️⃣ Preprocessing

- `preprocess_documents.py`
  - Converts text to lowercase
  - Removes non-alphabetic characters
  - Merges title, summary, and abstract into a single `clean_text` field

---

### 3️⃣ Indexing Engine

- `build_index.py` builds:
  - **Inverted Index** → term → document IDs + frequencies
  - **Positional Index** → term → document IDs + positions
  - **Document Metadata** → word counts, statistics

All indexes are stored as JSON files.

---

### 4️⃣ Search & Ranking

- Implemented in `app.py`
- Supports:
  - Vector Space Model (VSM)
  - TF-IDF weighting
  - Cosine Similarity
  - Boolean queries
  - Phrase queries using positional index

---

### 5️⃣ Web Interface

- Flask application
- Pages:
  - `index.html` → Search UI
  - `full_document.html` → Full document details
  - `error.html` → Error handling

Features:
- Highlighted query terms
- Relevance scores
- Pagination
- Direct links to PubMed


## 📁 Project Structure
```
MedSearch/
├── scripts/
│├───── generate_document_apis.py
│├───── preprocess_documents.py
│└───── build_index.py
│
├── web/
│├───── templates/
││├────────── index.html
││├────────── full_document.html
││└────────── error.html
││
│└───── app.py
│
└── requirements.txt
```

## 📜 License

#### **This project is for educational and research purposes.**


