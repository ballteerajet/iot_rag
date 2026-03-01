from firebase_client import get_sensor_data
from rag_engine import RAGEngine

def main():
    print("Loading sensor data from Firebase...")
    sensor_data = get_sensor_data()

    if not sensor_data:
        print("No data found.")
        return

    rag = RAGEngine()

    print("Converting sensor data...")
    texts = rag.convert_to_text(sensor_data)

    print("Building vector store...")
    rag.build_vector_store(texts)

    print("Chat system ready!\n")

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        context_docs = rag.retrieve(query)
        answer = rag.generate_answer(query, context_docs)

        print("\nAssistant:", answer)
        print("-" * 50)

if __name__ == "__main__":
    main()