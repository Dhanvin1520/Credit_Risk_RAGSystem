import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss


def _load_regulation_documents():
    docs = []
    reg_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'regulations')
    for filename in sorted(os.listdir(reg_dir)):
        if filename.endswith('.txt'):
            filepath = os.path.join(reg_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            source_name = filename.replace('.txt', '').replace('_', ' ').title()
            chunks = _chunk_text(content, chunk_size=300, overlap=50)
            for chunk in chunks:
                docs.append({'text': chunk, 'source': source_name})
    return docs


def _chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


class RegulationRetriever:
    def __init__(self):
        self.documents = _load_regulation_documents()
        texts = [doc['text'] for doc in self.documents]
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = self.vectorizer.fit_transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tfidf_matrix = tfidf_matrix / norms
        dimension = tfidf_matrix.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(tfidf_matrix)

    def retrieve(self, query: str, k: int = 5) -> list:
        query_vec = self.vectorizer.transform([query]).toarray().astype(np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        scores, indices = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({
                    'text': self.documents[idx]['text'],
                    'source': self.documents[idx]['source'],
                    'score': float(score)
                })
        return results
