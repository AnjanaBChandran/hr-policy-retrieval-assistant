# ingestion_pipeline.py

import os
import pickle
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = "data"
OUT_DIR = "tfidf_index"
CHUNK_SIZE = 500

os.makedirs(OUT_DIR, exist_ok=True)

def load_pdfs(folder):
    docs = []
    metadata = []

    for file in os.listdir(folder):
        if file.lower().endswith(".pdf"):
            path = os.path.join(folder, file)
            reader = PdfReader(path)

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue

                chunks = [
                    text[j : j + CHUNK_SIZE]
                    for j in range(0, len(text), CHUNK_SIZE)
                ]

                for chunk in chunks:
                    docs.append(chunk)
                    metadata.append({
                        "source": file,
                        "page": i + 1
                    })

    return docs, metadata


print("📄 Loading PDFs...")
documents, metadata = load_pdfs(DATA_DIR)

print(f"✅ Loaded {len(documents)} chunks")

print("🔢 Building TF-IDF index...")
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)
tfidf_matrix = vectorizer.fit_transform(documents)

# ✅ SAVE (PICKLE-SAFE)
with open(os.path.join(OUT_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(OUT_DIR, "tfidf_matrix.pkl"), "wb") as f:
    pickle.dump(tfidf_matrix, f)

with open(os.path.join(OUT_DIR, "metadata.pkl"), "wb") as f:
    pickle.dump({
        "documents": documents,
        "metadata": metadata
    }, f)

print("🎉 Ingestion complete. Files saved to tfidf_index/")
