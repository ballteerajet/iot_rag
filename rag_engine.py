import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

class RAGEngine:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.documents = []

    def convert_to_text(self, sensor_data):
        texts = []

        for key, values in sensor_data.items():

            if not isinstance(values, dict):
                continue

            temp = values.get("temp")
            moist = values.get("moist")
            timestamp = values.get("time")

            if temp is None or moist is None:
                continue

            try:
                temp = float(temp)
                moist = float(moist)
            except:
                continue

            description = "normal"
            if temp > 35:
                description = "very hot"
            elif temp > 30:
                description = "hot"
            elif temp < 20:
                description = "cold"

            text = (
                f"At timestamp {timestamp}, temperature was {temp} degrees Celsius "
                f"and soil moisture was {moist}. "
                f"The environment was {description}."
            )

            texts.append(text)

        return texts

    def build_vector_store(self, texts):
        self.documents = texts
        embeddings = self.embedder.encode(texts)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings))

    def retrieve(self, query, top_k=3):
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), top_k)

        results = [self.documents[i] for i in indices[0]]
        return results

    def generate_answer(self, query, context_docs):
        context = "\n".join(context_docs)

        prompt = f"""
    คุณคือผู้ช่วยวิเคราะห์สภาพแวดล้อมจากข้อมูลเซนเซอร์

    จงตอบเป็นภาษาไทย

    ข้อมูลจากเซนเซอร์:
    {context}

    คำถาม:
    {query}

    คำตอบ:
    """

        response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]