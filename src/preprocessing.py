from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class Preprocessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 500):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def data_loader(self):
        urls = [
            "https://docs.smoothbrained.co.uk/general/services/",
            "https://docs.smoothbrained.co.uk/general/roster/",
            "https://docs.smoothbrained.co.uk/general/deployment/",
            "https://docs.smoothbrained.co.uk/general/deployment/domain-names/",
            "https://docs.smoothbrained.co.uk/tutorials/adding-users/",
            "https://docs.smoothbrained.co.uk/tutorials/adding-machines/",
            "https://docs.smoothbrained.co.uk/tutorials/setting-up-a-webserver/",
            "https://docs.smoothbrained.co.uk/tutorials/vpn-guide/",
            "https://docs.smoothbrained.co.uk/tutorials/ldap/",
            "https://docs.smoothbrained.co.uk/tutorials/ldap/sys_accounts/",
        ]
        
        docs = [WebBaseLoader(url).load() for url in urls]
        return docs
    
    def data_splitter(self, docs):
        docs_list = [item for sublist in docs for item in sublist]
        doc_splits = self.text_splitter.split_documents(docs_list)
        return doc_splits
 
