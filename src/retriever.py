# src/retriever.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.texts = []

    def build_index(self, chunks):
        embeddings = self.embedder.encode(chunks, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.texts = chunks
        print("Indices built")

    def retrieve(self, query, k=3):
        query_emb = self.embedder.encode([query])
        distances, indices = self.index.search(np.array(query_emb, dtype=np.float32), k)
        print("Retrieving")
        return [self.texts[i] for i in indices[0]]

