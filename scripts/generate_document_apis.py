import requests
import xml.etree.ElementTree as ET
import json
import time
import os

DISEASES = [
    "diabetes mellitus",
    "hypertension",
    "coronary artery disease",
    "asthma",
    "chronic obstructive pulmonary disease",
    "stroke",
    "breast cancer",
    "lung cancer",
    "colorectal cancer",
    "alzheimer disease",
    "parkinson disease",
    "rheumatoid arthritis",
    "osteoarthritis",
    "chronic kidney disease",
    "liver cirrhosis",
    "tuberculosis",
    "pneumonia",
    "covid-19",
    "influenza",
    "epilepsy"
]

ARTICLES_PER_DISEASE = 500
BATCH_SIZE = 20
OUTPUT_DIR = "pubmed_articles"

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


# -----------------------------
def search_pubmed(term, retstart, retmax):
    params = {
        "db": "pubmed",
        "term": term,
        "retstart": retstart,
        "retmax": retmax,
        "retmode": "json"
    }
    r = requests.get(ESEARCH_URL, params=params)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def fetch_articles(pmids):
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml"
    }
    r = requests.get(EFETCH_URL, params=params)
    r.raise_for_status()
    return r.text


def parse_articles(xml_data, disease):
    root = ET.fromstring(xml_data)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        title = article.findtext(".//ArticleTitle")

        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join(
            [p.text for p in abstract_parts if p.text]
        )

        if pmid and title and abstract:
            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "disease": disease
            })

    return articles


# -----------------------------
def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    seen_pmids = set()
    counter = 9278

    for disease in DISEASES:
        print(f"\n=== Fetching: {disease} ===")

        for start in range(0, ARTICLES_PER_DISEASE, BATCH_SIZE):
            pmids = search_pubmed(disease, start, BATCH_SIZE)
            if not pmids:
                break

            xml_data = fetch_articles(pmids)
            articles = parse_articles(xml_data, disease)

            for article in articles:
                if article["pmid"] not in seen_pmids:
                    seen_pmids.add(article["pmid"])
                    
                    # create filename like data1.json, data2.json...
                    filename = f"data{counter}.json"
                    filepath = os.path.join(OUTPUT_DIR, filename)

                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(article, f, ensure_ascii=False, indent=2)

                    counter += 1

            time.sleep(0.4)

        print(f"Saved so far: {counter - 1}")

    print(f"\nDONE ✅ Total unique files created: {counter - 1}")


if __name__ == "__main__":
    main()
