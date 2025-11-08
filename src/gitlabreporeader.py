import os
from git import Repo
from typing import List


class GitlabRepoReader:
    def __init__(self, repo_url: str, local_dir: str = "./repo"):
        self.repo_url = repo_url
        self.local_dir = local_dir

    def clone_repo(self, force_update: bool = False) -> None:
        if not os.path.exists(self.local_dir):
            print(f"Cloning repository from {self.repo_url}...")
            Repo.clone_from(self.repo_url, self.local_dir)
            print("✅ Repository cloned successfully.")
        else:
            repo = Repo(self.local_dir)
            if force_update:
                print("Pulling latest changes...")
                repo.remotes.origin.pull()
                print("✅ Repository updated.")
            else:
                print("Repository already exists. Skipping clone.")

    def get_markdown_files(self) -> List[str]:
        markdown_files = []
        for root, _, files in os.walk(self.local_dir):
            for file in files:
                if file.lower().endswith(".md"):
                    markdown_files.append(os.path.join(root, file))
        print(f"📄 Found {len(markdown_files)} markdown files.")
        return markdown_files
