from src import Preprocessor
from src.retriever import Retriever
#from src.model_interface import OllamaModel
 
def main():
    preprocessor = Preprocessor()
    texts = preprocessor.data_loader()
    chunks = preprocessor.data_splitter(texts)

    retriever = Retriever()
    retriever.build_index(chunks)

 #   query = "What are the key insights from the text?"
 #   retrieved_chunks = retriever.retrieve(query)

 #   model = OllamaModel()
 #   context = "\n".join(retrieved_chunks)
 #   prompt = f"Use the context below to answer the question:\n\n{context}\n\nQuestion: {query}"
 #   answer = model.generate(prompt)
 #   print("💬 Llama3.1 answer:\n", answer)


if __name__ == "__main__":
    main()
