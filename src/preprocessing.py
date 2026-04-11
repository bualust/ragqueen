from collections import deque
from urllib.parse import urljoin, urlparse, urldefrag
import requests
from bs4 import BeautifulSoup

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

    def _normalize_url(self, url: str) -> str:
        """Remove fragments and normalize trailing slash."""
        url, _ = urldefrag(url)  
        return url

    def _is_same_domain_or_subdomain(self, url: str, base_netloc: str) -> bool:
        """
        True if url belongs to the same domain or a subdomain of base_netloc.
        Example:
            base_netloc = example.com
            allows:
                example.com
                www.example.com
                docs.example.com
        """
        try:
            parsed = urlparse(url)
            netloc = parsed.netloc.lower()
            base_netloc = base_netloc.lower()

            return netloc == base_netloc or netloc.endswith("." + base_netloc)
        except Exception:
            return False

    def _extract_clean_text(self, html: str) -> str:
        """Extract visible text from HTML."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove script/style/noscript content
        # which is the no-human readable part
        # of a website
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        return "\n".join(lines)

    def crawl_site(
        self,
        base_url: str,
        max_pages: int = 100,
        max_depth: int = 4,
        include_subdomains: bool = True,
        timeout: int = 10,
        user_agent: str = "Mozilla/5.0 (compatible; PreprocessorBot/1.0)"
    ) -> list[Document]:
        """
        Crawl a site starting from base_url and return a list of Documents.

        Notes:
        - This crawls linked pages only.
        - It does NOT enumerate all existing subdomains unless they are linked.
        """
        session = requests.Session()
        session.headers.update({"User-Agent": user_agent})

        parsed_base = urlparse(base_url)
        if not parsed_base.scheme:
            raise ValueError("base_url must include scheme, e.g. https://example.com")

        base_netloc = parsed_base.netloc.lower()
        base_url = self._normalize_url(base_url)

        visited = set()
        queue = deque([(base_url, 0)])
        docs = []

        while queue and len(visited) < max_pages:
            current_url, depth = queue.popleft()
            current_url = self._normalize_url(current_url)

            if current_url in visited:
                continue

            if depth > max_depth:
                continue

            try:
                print(f"Crawling [{len(visited)+1}/{max_pages}] depth={depth}: {current_url}")
                response = session.get(current_url, timeout=timeout)
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" not in content_type:
                    # Skip PDFs/images/etc. in this crawler
                    visited.add(current_url)
                    continue

                html = response.text
                text = self._extract_clean_text(html)

                if text.strip():
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": current_url,
                                "depth": depth,
                                "content_type": content_type,
                            }
                        )
                    )

                visited.add(current_url)

                # Parse links
                soup = BeautifulSoup(html, "html.parser")
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"].strip()

                    # Skip mailto:, javascript:, tel:
                    if href.startswith(("mailto:", "javascript:", "tel:")):
                        continue

                    absolute_url = urljoin(current_url, href)
                    absolute_url = self._normalize_url(absolute_url)
                    parsed = urlparse(absolute_url)

                    # Only http(s)
                    if parsed.scheme not in ("http", "https"):
                        continue

                    # Domain filtering
                    if include_subdomains:
                        allowed = self._is_same_domain_or_subdomain(absolute_url, base_netloc)
                    else:
                        allowed = parsed.netloc.lower() == base_netloc

                    if not allowed:
                        continue

                    if absolute_url not in visited:
                        queue.append((absolute_url, depth + 1))

            except requests.RequestException as e:
                print(f"Failed to fetch {current_url}: {e}")
                visited.add(current_url)
                continue
            except Exception as e:
                print(f"Error processing {current_url}: {e}")
                visited.add(current_url)
                continue

        return docs

    def data_loader_urls(self, urls) -> list[Document]:
        """
        Load pages from:
        - a list of URLs, OR
        - a single base URL to crawl

        If `urls` is a string, treat it as a base URL and crawl it.
        If `urls` is a list, fetch each URL directly without crawling links.
        """
        if isinstance(urls, str):
            print(f"Crawling base site: {urls}")
            return self.crawl_site(urls)

        elif isinstance(urls, list):
            print(f"Loading specific URLs:\n{urls}")
            docs = []
            session = requests.Session()
            session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; PreprocessorBot/1.0)"})

            for url in urls:
                try:
                    response = session.get(url, timeout=10)
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "").lower()

                    if "text/html" in content_type:
                        text = self._extract_clean_text(response.text)
                        docs.append(
                            Document(
                                page_content=text,
                                metadata={"source": url, "content_type": content_type}
                            )
                        )
                except requests.RequestException as e:
                    print(f"Failed to fetch {url}: {e}")

            return docs

        else:
            raise ValueError("`urls` must be either a string (base URL) or a list of URLs.")

    def data_loader_repo(self, repo_url) -> list[Document]:
        """Load MarkDown files from git repository"""
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
        """
        Check if a repo was provided or URLs/base website and return docs.
        """
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
        """Flatten nested lists if needed, then split docs."""
        if any(isinstance(d, list) for d in docs):
            docs_list = [item for sublist in docs for item in sublist]
        else:
            docs_list = docs
        return self.text_splitter.split_documents(docs_list)
