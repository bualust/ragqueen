from src.retriever import Retriever
from src.model_interface import OllamaModel
 
def main():

    retriever = Retriever()
    retriever.load("index_store")
    query = input("Enter your question: ")
    retrieved_chunks = retriever.retrieve(query)

    model = OllamaModel()
    context = "\n".join(retrieved_chunks)
    prompt = f"Use the context below to answer the question:\n\n{context}\n\nQuestion: {query}"
    answer = model.generate(prompt)
    print("💬 Llama3.1 answer:\n", answer)

if __name__ == "__main__":
    main()
