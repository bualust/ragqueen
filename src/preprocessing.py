from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from src.gitlabreporeader import GitlabRepoReader

class Preprocessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 500):
        print(f"Using chunk_size = {chunk_size}, chunk_overlap = {chunk_overlap}")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def data_loader_urls(self, urls) -> list[Document]:
        print(f"Scanning following urls:\n{urls}")
        docs = [WebBaseLoader(url).load() for url in urls]
        return docs
    
    def data_loader_repo(self, repo_url) -> list[Document]:
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

    def loaded_list(self, yaml_data) -> list[Document]:
        """ checks if a repo was provided or a
        list of urls and return texts list """

        has_urls = "urls" in yaml_data and yaml_data["urls"]
        has_repo = "repo_url" in yaml_data and yaml_data["repo_url"]

        if has_urls and has_repo:
            raise ValueError("Only provide either `urls` or `repo_url`")
        elif has_urls:
            return self.data_loader_urls(yaml_data["urls"])
        elif has_repo:
            return self.data_loader_repo(yaml_data["repo_url"])
        else:
            raise ValueError("Provide either `urls` or `repo_url`")

    def data_splitter(self, docs):
        if any(isinstance(d, list) for d in docs):
            docs_list = [item for sublist in docs for item in sublist]
        else:
            docs_list = docs
        return self.text_splitter.split_documents(docs_list) 
