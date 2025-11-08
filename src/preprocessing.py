from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from src.gitlabreporeader import GitlabRepoReader

class Preprocessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 500):
        print(f"Using chunk_size = {chunk_size}, chunk_overlap = {chunk_overlap}")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def data_loader_urls(self,urls):
        print(f"Scanning following urls:\n{urls}")
        docs = [WebBaseLoader(url).load() for url in urls]
        return docs
    
    def data_loader_repo(self,repo_url):
        repo_reader = GitlabRepoReader(
            repo_url,
            local_dir="./repo_cache"
        )
        repo_reader.clone_repo(force_update=False)
        md_files = repo_reader.get_markdown_files()
        docs = []
        for md_path in md_files:
            loader = UnstructuredMarkdownLoader(str(md_path), mode="elements")
            docs.extend(loader.load())
        return docs

    def data_splitter(self, docs):
        if any(isinstance(d, list) for d in docs):
            docs_list = [item for sublist in docs for item in sublist]
        else:
            docs_list = docs
        return self.text_splitter.split_documents(docs_list) 
