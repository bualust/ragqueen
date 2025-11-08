from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Preprocessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 500):
        print(f"Using chunk_size = {chunk_size}, chunk_overlap = {chunk_overlap}")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def data_loader(self,urls):
        print(f"Scanning following urls:\n{urls}")
        docs = [WebBaseLoader(url).load() for url in urls]
        return docs
    
    def data_splitter(self, docs):
        docs_list = [item for sublist in docs for item in sublist]
        doc_splits = self.text_splitter.split_documents(docs_list)
        return doc_splits
 
